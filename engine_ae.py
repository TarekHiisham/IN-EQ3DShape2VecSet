# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import e3nn.o3 as o3

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, criterion_lat: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    preservation = args.preservation

    optimizer.zero_grad()

    kl_weight = 1e-3

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        
        outputs_rot = None
        loss_vol_pres = None
        loss_near_pres = None
        loss_pres = None
        iou_rot = None

        with torch.amp.autocast('cuda', enabled=False):
            o = model(surface, points)
            
            if 'kl' in o:
                loss_kl = o['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = o['logits']

            loss_vol = criterion(outputs[:, :1024], labels[:, :1024])
            loss_near = criterion(outputs[:, 1024:], labels[:, 1024:])
            
            if loss_kl is not None:
                loss = loss_vol + 0.1 * loss_near + kl_weight * loss_kl 
            else:
                loss = loss_vol + 0.1 * loss_near

            if preservation == 'inv':
                R = o3.rand_matrix(points.shape[0], dtype=points.dtype)
                R = R.to(device)
                points_rot = torch.einsum('bij, bnj -> bni', R, points)
                surface_rot = torch.einsum('bij, bnj -> bni', R, surface)
                
                o_rot = model(surface_rot, points_rot)
                outputs_rot = o_rot['logits']

                loss_vol_pres = criterion(outputs_rot[:, :1024], labels[:, :1024])
                loss_near_pres = criterion(outputs_rot[:, 1024:], labels[:, 1024:])
                loss_pres = loss_vol_pres + 0.1 * loss_near_pres
                loss = loss + loss_pres

            elif preservation == 'equ':
                R = o3.rand_matrix(points.shape[0], dtype=points.dtype)
                irreps = o3.Irreps("128x0e + 128x1o")
                D = irreps.D_from_matrix(R)
                R = R.to(device)
                D = D.to(device)

                points_rot = torch.einsum('bij, bnj -> bni', R, points)
                surface_rot = torch.einsum('bij, bnj -> bni', R, surface)

                o_rot = model(surface_rot, points_rot)
                latents_rot = o_rot['latents']
                outputs_rot = o_rot['logits']

                lat_feat_expected = torch.einsum('bij, bnj -> bni', D, o['latents'])

                loss_pres = criterion_lat(latents_rot, lat_feat_expected)
                loss = loss + loss_pres

            else:
                outputs_rot = None
                loss_vol_pres = None
                loss_near_pres = None
                loss_pres = None
                iou_rot = None


        loss_value = loss.item()

        threshold = 0

        pred = torch.zeros_like(outputs[:, :1024])
        pred[outputs[:, :1024]>=threshold] = 1

        accuracy = (pred==labels[:, :1024]).float().sum(dim=1) / labels[:, :1024].shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels[:, :1024]).sum(dim=1)
        union = (pred + labels[:, :1024]).gt(0).sum(dim=1) + 1e-5
        iou = intersection * 1.0 / union
        iou = iou.mean()
        
        if outputs_rot is not None:
            pred_rot = torch.zeros_like(outputs_rot[:, :1024])
            pred_rot[outputs_rot[:, :1024]>=threshold] = 1

            accuracy_rot = (pred_rot==labels[:, :1024]).float().sum(dim=1) / labels[:, :1024].shape[1]
            accuracy_rot = accuracy_rot.mean()
            intersection_rot = (pred_rot * labels[:, :1024]).sum(dim=1)
            union_rot = (pred_rot + labels[:, :1024]).gt(0).sum(dim=1) + 1e-5
            iou_rot = intersection_rot * 1.0 / union_rot
            iou_rot = iou_rot.mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())

        if loss_vol_pres is not None:
            metric_logger.update(loss_vol_inv=loss_vol_pres.item())
            metric_logger.update(loss_near_inv=loss_near_pres.item())

        if loss_pres is not None:
            metric_logger.update(loss_equ_lat=loss_pres.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        metric_logger.update(iou=iou.item())
        
        if iou_rot is not None:
            metric_logger.update(iou_rot=iou_rot.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    preservation = args.preservation

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_lat = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for points, labels, surface in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        o_rot = None
        iou_rot = None
        loss_pres = None
        
        # compute output
        with torch.amp.autocast('cuda', enabled=False):

            o = model(surface, points)

            if 'kl' in o:
                loss_kl = o['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = o['logits']

            loss = criterion(outputs, labels)

            if preservation == 'inv':
                R = o3.rand_matrix(points.shape[0], dtype=points.dtype)
                R = R.to(device)
                points_rot = torch.einsum('bij, bnj -> bni', R, points)
                surface_rot = torch.einsum('bij, bnj -> bni', R, surface)
                
                o_rot = model(surface_rot, points_rot)
                outputs_rot = o_rot['logits']

                loss_pres = criterion(outputs_rot, labels)

                loss = loss + loss_pres 

            elif preservation == 'equ':
                R = o3.rand_matrix(points.shape[0], dtype=points.dtype)
                irreps = o3.Irreps("128x0e + 128x1o")
                D = irreps.D_from_matrix(R)
                R = R.to(device)
                D = D.to(device)

                points_rot = torch.einsum('bij, bnj -> bni', R, points)
                surface_rot = torch.einsum('bij, bnj -> bni', R, surface)

                o_rot = model(surface_rot, points_rot)
                latents_rot = o_rot['latents']
                outputs_rot = o_rot['logits']

                lat_feat_expected = torch.einsum('bij, bnj -> bni', D, o['latents'])

                loss_pres = criterion_lat(latents_rot, lat_feat_expected)
                loss = loss + loss_pres

            else:
                o_rot = None
                iou_rot = None
                loss_pres = None

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()

        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        if o_rot is not None:
            pred_rot = torch.zeros_like(outputs_rot)
            pred_rot[outputs_rot>=threshold] = 1

            accuracy_rot = (pred_rot==labels).float().sum(dim=1) / labels.shape[1]
            accuracy_rot = accuracy_rot.mean()
            intersection_rot = (pred_rot * labels).sum(dim=1)
            union_rot = (pred_rot + labels).gt(0).sum(dim=1)
            iou_rot = intersection_rot * 1.0 / union_rot + 1e-5
            iou_rot = iou_rot.mean()
                    
        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item()) 

        if loss_pres is not None:
            metric_logger.update(loss_pres=loss_pres.item())

        metric_logger.meters['iou'].update(iou.item(), n=batch_size)
        if iou_rot is not None:
            metric_logger.meters['iou_rot'].update(iou_rot.item(), n=batch_size)

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} iou_rot {iou_rot.global_avg:.3f} loss {losses.global_avg:.3f} loss_pres {losses_pres.global_avg:.7f}'
          .format(iou=metric_logger.iou, iou_rot=metric_logger.iou_rot, losses=metric_logger.loss, losses_pres=metric_logger.loss_pres))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}