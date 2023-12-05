![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Global Vision Transformer Pruning with Hessian-Aware Saliency

This repository is the official PyTorch implementation of [Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869) (Also known as NViT) presented at CVPR 2023.

---

## ReViT training from scratch

Our work observes a unique parameter redistribution trend from the dimensions of the pruned model, and propose to use it for designing efficient architecture. In this section, we present ReViT repository to evaluate our insight on ViT parameter redistribution, and to provide a tool for users to flexibly explore novel ViT designs with different dimensions in each block.


The following command explores the effectiveness of training the redistributed model from scratch.

ReViT-T - Hardware: 8 V100 (32G) NVIDIA GPUs

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env revit_scratch.py --model deit_base_distilled_patch16_224 --epochs 300 --num_workers 4 --batch-size 32 --data-path /path/to/ImageNet2012/ --data-set IMNET --output_dir save/path --amp --input-size 224 --seed 1 --kl_loss_coeff=0 --original_loss_coeff=1.0 --dist-eval --pretrained --distillation-type hard --distillation-alpha 0.5 --distillation-tau 1.0 --lr 0.0005 --scratch --EMB 176 --warmup-epochs 5
```

ReViT-S - Hardware: 8 V100 (32G) NVIDIA GPUs

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env revit_scratch.py --model deit_base_distilled_patch16_224 --epochs 300 --num_workers 4 --batch-size 32 --data-path /path/to/ImageNet2012/ --data-set IMNET --output_dir save/path --amp --input-size 224 --seed 1 --kl_loss_coeff=0 --original_loss_coeff=1.0 --dist-eval --pretrained --distillation-type hard --distillation-alpha 0.5 --distillation-tau 1.0 --lr 0.0005 --scratch --EMB 384 --warmup-epochs 5
```

Arguments:

- `epochs` - number of training epochs
- `data-path` - path to ImageNet dataset
- `output_dir` - path to save training log and the final model. Important arguments will be automatically appended to the specified folder name
- `lr` - initial learning rate, cosine decay is used for the entire training
- `warmup-epochs` - epochs for linear learning rate warmup
- `EMB` - embedding size for scaling the model architecture

You can also use this code to freely explore your own design of the redistributed architecture. Line 366 - 369 of revit_scratch.py can be modifed to freely adjust the dimension of each independent structural component in a DeiT model. 


## Licenses

Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](../LICENSE) to view a copy of this license.

For license information regarding the DeiT repository, please refer to its [repository](https://github.com/facebookresearch/deit).


## Acknowledgement


This repository is built on top of the [timm](https://github.com/huggingface/pytorch-image-models) repository. We thank [Ross Wrightman](https://rwightman.com/) for creating and maintaining this high-quality library.  

Part of this code is modified from the official repo of [DeiT](https://github.com/facebookresearch/deit.git). We thank the authors for their amazing work and releasing their code base. 
