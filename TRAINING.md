![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# Training Instructions

In this section, we provide exact pruning and finetuning details of all NViT models. We use a single node of 4 x V100 NVIDIA GPUs for pruning and 4 nodes (8 x V100 NVIDIA GPUs) for finrtuning all models, unless otherwise specified. Training and finetuning codes are available in the `nvit` folder.



## Pretrained model pruning

`main_full_global_latency.py` can be used to perform global structural pruning on a pretrained DeiT-B model towards a target speedup with the following arguments. The pretrained DeiT-B model checkpoint will be downloaded automatically.

Arguments:

- `epochs` - maximum epochs of pruning, can be set to a large number, code will stop when target latency is reached
- `data-path` - path to ImageNet dataset
- `output_dir` - path to save pruning log and final pruned model. Important arguments will be automatically appended to the specified folder name
- `lr` - learning rate for model update during iterative pruning
- `prune_per_iter` - number of neurons pruned in each pruning iteration
- `interval_prune` - number of optimization steps (batches) within each pruning iteration
- `latency_regularization` - weight of latency regularization in overall sensitivity 
- `latency_target` - ratio of the target latency of the final pruned model and the latency of the original model

The pruning process will stop after the target latency is reached. The model after pruning is stored in the `pruned_checkpoint.pth`, and the pruning log and remaining dimension of each pruning step can be found in the `debug` folder.

NViT-B

```
 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /path/to/ImageNet2012/ --data-set IMNET --lr 1e-4 --output_dir save/path --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.54 --latency_look_up_table latency_head.json --pruning_exit
```

NViT-H

```
 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /path/to/ImageNet2012/ --data-set IMNET --lr 1e-4 --output_dir save/path --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.50 --latency_look_up_table latency_head.json --pruning_exit
```

NViT-S

```
 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /path/to/ImageNet2012/ --data-set IMNET --lr 1e-4 --output_dir save/path --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.39 --latency_look_up_table latency_head.json --pruning_exit
```

NViT-T

```
 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /path/to/ImageNet2012/ --data-set IMNET --lr 1e-4 --output_dir save/path --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.19 --latency_look_up_table latency_head.json --pruning_exit
```



## Pruned model finetune and train from scratch

After pruning, `finetune_dense.py` can be used to convert the pruned model into a small dense model, and perform finetuning. The code also supports training a model with the same architecture as the pruned model from scratch.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune_dense.py --model deit_base_distilled_patch16_224 --epochs 300 --num_workers 10 --batch-size 144 --data-path /path/to/ImageNet2012/ --data-set IMNET --amp --input-size 224 --seed 1 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --dist-eval --pretrained --finetune path/to/pruned_model --distillation-type hard --distillation-alpha 0.5 --distillation-tau 20.0 --lr 0.0002
```

Arguments:

- `epochs` - number of finetuning epochs
- `data-path` - path to ImageNet dataset
- `finetune` - path to the folder containing `pruned_checkpoint.pth`
- `lr` - initial learning rate, cosine decay is used for the entire training
- `kl_loss_coeff` - weight of full model distillation objective in the overall training loss, set to 0 to use CNN distillation only
- `distillation-type` - CNN distillation type, same as in DeiT
- `distillation-tau` - tempreture of full model distillation

Add `--scratch` to the command to train a model with the same architecture as the pruned model from scratch, instead of finetuning from the pruned checkpoint.

The training log and the final finetuned model will be stored in a new folder generated in the path specified by `--finetune`. Note that the resulted ft_checkpoint.pth will be significantly smaller than the original pruned_checkpoint.pth as the all-zero dimensions in the pruned model are removed.

The finetuning process can be resumed if it's stopped accidentally midway. To resume a stopped job, check the current epoch the model is in from the log, and add `--resume --start_epoch {epoch}` to the original command. The code will automatically find the correct log folder and resume the tuning. DO NOT change any hyperparameter from the stopped command when resuming.

## Data Preparation

Please download the ImageNet dataset from its official website. The training and validation images need to have
sub-folders for each class with the following structure:

```bash
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```



