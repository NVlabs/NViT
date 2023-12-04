import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
from apex import amp

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def finetune_one_epoch(args, student, teacher, masks, criterion, orig_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,main_loss_coeff=100000,original_loss_coeff=0.):
    student.train(set_training_mode)
    teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('main_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('orig_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            # get student predictions
            student_logits = student(samples)
            
            if main_loss_coeff>0:
                with torch.no_grad():
                    teacher_logits = teacher(samples)
                # get distillation loss
                main_loss = (criterion(student_logits[0], teacher_logits[0])+criterion(student_logits[1], teacher_logits[1]))/2
            else:
                main_loss = 0.
            
            original_loss_student = orig_criterion(samples, student_logits, targets)

        #total loss
        loss = main_loss_coeff*main_loss + original_loss_coeff*original_loss_student
        loss_value = loss.item()
        main_loss_value = 0.
        if main_loss_coeff>0:
            main_loss_value = main_loss.item()
        orig_loss_value = original_loss_student.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=student.parameters(), create_graph=is_second_order)

        if masks is not None:
            for name, p in student.named_parameters():
                if 'weight' in name or 'bias' in name or 'mask' in name:
                    p.grad.data = p.grad.data*masks[name]
                    p.data = p.data*masks[name]

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(student)

        metric_logger.update(loss=loss_value)
        metric_logger.update(main_loss_value=main_loss_value)
        metric_logger.update(orig_loss_value=orig_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            x, x_dist = model(images)
            output = (x + x_dist) / 2
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
