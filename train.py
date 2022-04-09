import collections
import math
import statistics
from functools import reduce

import torch
import torch.nn as nn
from apex import amp
from torch import distributed
from torch.nn import functional as F

from utils import get_regularizer
from utils.loss import (NCA, BCESigmoid, BCEWithLogitsLossWithIgnoreIndex,
                        ExcludedKnowledgeDistillationLoss, FocalLoss,
                        FocalLossNew, IcarlLoss, KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss, UnbiasedNCA,
                        soft_crossentropy)

def features_distillation8( 
    list_attentions_a,
    list_attentions_b,
    index_new_class=16,
    nb_current_classes=16,
    nb_new_classes=1
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    list_attentions_a = list_attentions_a[:-1]
    list_attentions_b = list_attentions_b[:-1]
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        layer_loss = torch.tensor(0.).to(a.device)
    
        if a.shape[1] != b.shape[1]:
            _b = torch.zeros_like(a).to(a.dtype).to(a.device)
            _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
            _b[:, 1:] = b[:, 1:index_new_class]
            b = _b

        a = a ** 2
        b = b ** 2
        a_affinity_4 = F.avg_pool2d(a, (4, 4), stride=1, padding=2)
        b_affinity_4 = F.avg_pool2d(b, (4, 4), stride=1, padding=2)
        a_affinity_8 = F.avg_pool2d(a, (8, 8), stride=1, padding=4)
        b_affinity_8 = F.avg_pool2d(b, (8, 8), stride=1, padding=4)
        a_affinity_12 = F.avg_pool2d(a, (12, 12), stride=1, padding=6)
        b_affinity_12 = F.avg_pool2d(b, (12, 12), stride=1, padding=6)
        a_affinity_16 = F.avg_pool2d(a, (16, 16), stride=1, padding=8)
        b_affinity_16 = F.avg_pool2d(b, (16, 16), stride=1, padding=8)
        a_affinity_20 = F.avg_pool2d(a, (20, 20), stride=1, padding=10)
        b_affinity_20 = F.avg_pool2d(b, (20, 20), stride=1, padding=10)
        a_affinity_24 = F.avg_pool2d(a, (24, 24), stride=1, padding=12)
        b_affinity_24 = F.avg_pool2d(b, (24, 24), stride=1, padding=12)

        layer_loss = torch.frobenius_norm((a_affinity_4 - b_affinity_4).view(a.shape[0], -1), dim=-1).mean() + \
                    torch.frobenius_norm((a_affinity_8 - b_affinity_8).view(a.shape[0], -1), dim=-1).mean() + \
                    torch.frobenius_norm((a_affinity_16 - b_affinity_16).view(a.shape[0], -1), dim=-1).mean() + \
                    torch.frobenius_norm((a_affinity_20 - b_affinity_20).view(a.shape[0], -1), dim=-1).mean() + \
                    torch.frobenius_norm((a_affinity_24 - b_affinity_24).view(a.shape[0], -1), dim=-1).mean() + \
                    torch.frobenius_norm((a_affinity_12 - b_affinity_12).view(a.shape[0], -1), dim=-1).mean()
        layer_loss = layer_loss / 6.
            

        if i == len(list_attentions_a) - 1:
            # pod_factor = 0.0001 # cityscapes
            pod_factor = 0.0005 # voc
        else:
            # pod_factor = 0.0005 # cityscapes
            pod_factor = 0.01 # voc
      
        loss = loss + layer_loss.mean() * 1.0 * math.sqrt(nb_current_classes / nb_new_classes) * pod_factor


    return loss / len(list_attentions_a)


def features_distillation_channel( 
    list_attentions_a,
    list_attentions_b,
    index_new_class=16,
    nb_current_classes=16,
    nb_new_classes=1
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    list_attentions_a = list_attentions_a[:-2]
    list_attentions_b = list_attentions_b[:-2]
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        layer_loss = torch.tensor(0.).to(a.device)
    
        if a.shape[1] != b.shape[1]:
            _b = torch.zeros_like(a).to(a.dtype).to(a.device)
            _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
            _b[:, 1:] = b[:, 1:index_new_class]
            b = _b

        a = a ** 2
        b = b ** 2

        a_p = F.avg_pool2d(a.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
        b_p = F.avg_pool2d(b.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
        
        layer_loss = torch.frobenius_norm((a_p - b_p).view(a.shape[0], -1), dim=-1).mean()

        if i == len(list_attentions_a) - 1:
            pod_factor = 0.0
        else:
            pod_factor = 0.0001 # cityscapes
        pod_factor = 0.01 #voc
        loss = loss + layer_loss.mean() * 1.0 * math.sqrt(nb_current_classes / nb_new_classes) * pod_factor


    return loss / len(list_attentions_a)


class Trainer:

    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, step=0):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.step = step
        self.opts = opts
        if opts.dataset == "cityscapes_domain":
            if self.step > 0:
                self.old_classes = opts.num_classes
                self.nb_classes = opts.num_classes
            else:
                self.old_classes = 0
                self.nb_classes = None
            self.nb_current_classes = opts.num_classes
            self.nb_new_classes = opts.num_classes
        elif classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0
            self.nb_classes = None

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        elif opts.nca and self.old_classes != 0:
            self.criterion = UnbiasedNCA(
                old_cl=self.old_classes,
                ignore_index=255,
                reduction=reduction,
                scale=model.module.scalar,
                margin=opts.nca_margin
            )
        elif opts.nca:
            self.criterion = NCA(
                scale=model.module.scalar,
                margin=opts.nca_margin,
                ignore_index=255,
                reduction=reduction
            )
        elif opts.focal_loss:
            self.criterion = FocalLoss(ignore_index=255, reduction=reduction, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        elif opts.focal_loss_new:
            self.criterion = FocalLossNew(ignore_index=255, reduction=reduction, index=self.old_classes, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_mask = opts.kd_mask
        self.kd_mask_adaptative_factor = opts.kd_mask_adaptative_factor
        self.lkd_flag = self.lkd > 0. and model_old is not None
        self.kd_need_labels = False
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.kd_bce_sig:
            self.lkd_loss = BCESigmoid(reduction="none", alpha=opts.alpha, shape=opts.kd_bce_sig_shape)
        elif opts.exkd_gt and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="gt",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        elif opts.exkd_sum and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="sum",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or (opts.pod is not None)

        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        self.step_threshold = opts.step_threshold
        self.ce_on_pseudo = opts.ce_on_pseudo
        self.pseudo_nb_bins = opts.pseudo_nb_bins
        self.pseudo_soft = opts.pseudo_soft
        self.pseudo_soft_factor = opts.pseudo_soft_factor
        self.pseudo_ablation = opts.pseudo_ablation
        self.classif_adaptive_factor = opts.classif_adaptive_factor
        self.classif_adaptive_min_factor = opts.classif_adaptive_min_factor

        self.kd_new = opts.kd_new
        self.pod = opts.pod
        self.pod_options = opts.pod_options if opts.pod_options is not None else {}
        self.pod_factor = opts.pod_factor
        self.pod_prepro = opts.pod_prepro
        self.use_pod_schedule = not opts.no_pod_schedule
        self.pod_deeplab_mask = opts.pod_deeplab_mask
        self.pod_deeplab_mask_factor = opts.pod_deeplab_mask_factor
        self.pod_apply = opts.pod_apply
        self.pod_interpolate_last = opts.pod_interpolate_last
        self.deeplab_mask_downscale = opts.deeplab_mask_downscale
        self.spp_scales = opts.spp_scales
        self.pod_logits = opts.pod_logits
        self.pod_large_logits = opts.pod_large_logits

        self.align_weight = opts.align_weight
        self.align_weight_frequency = opts.align_weight_frequency

        self.dataset = opts.dataset

        self.entropy_min = opts.entropy_min

        self.kd_scheduling = opts.kd_scheduling

        self.sample_weights_new = opts.sample_weights_new

        self.temperature_apply = opts.temperature_apply
        self.temperature = opts.temperature

        # CIL
        self.ce_on_new = opts.ce_on_new

    def before(self, train_loader, logger):
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, _ = self.find_median(train_loader, self.device, logger)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.device, logger, mode="entropy"
            )

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info(f"Pseudo labeling is: {self.pseudo_labeling}")
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
         

        device = self.device
        model = self.model
        criterion = self.criterion

        model.module.in_eval = False
        if self.model_old is not None:
            self.model_old.in_eval = False

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        pod_loss = torch.tensor(0.)
        loss_entmin = torch.tensor(0.)

        sample_weights = None

        train_loader.sampler.set_epoch(cur_epoch)


        for cur_step, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            original_labels = labels.clone()

            if (
                self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.pod is not None or
                self.pseudo_labeling is not None
            ) and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )

            classif_adaptive_factor = 1.0
            if self.step > 0:
                mask_background = labels < self.old_classes

                if self.pseudo_labeling == "naive":
                    labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
                elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith(
                    "threshold_"
                ):
                    threshold = float(self.pseudo_labeling.split("_")[1])
                    probs = torch.softmax(outputs_old, dim=1)
                    pseudo_labels = probs.argmax(dim=1)
                    pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "confidence":
                    probs_old = torch.softmax(outputs_old, dim=1)
                    labels[mask_background] = probs_old.argmax(dim=1)[mask_background]
                    sample_weights = torch.ones_like(labels).to(device, dtype=torch.float32)
                    sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
                elif self.pseudo_labeling == "median":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "entropy":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)

                    mask_valid_pseudo = (entropy(probs) /
                                         self.max_entropy) < self.thresholds[pseudo_labels]


                    if self.pseudo_soft is None:
                        # All old labels that are NOT confident enough to be used as pseudo labels:
                        labels[~mask_valid_pseudo & mask_background] = 255

                        if self.pseudo_ablation is None:
                            # All old labels that are confident enough to be used as pseudo labels:
                            labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]
                        elif self.pseudo_ablation == "corrected_errors":
                            pass  # If used jointly with data_masking=current+old, the labels already
                                  # contrain the GT, thus all potentials errors were corrected.
                        elif self.pseudo_ablation == "removed_errors":
                            pseudo_error_mask = labels != pseudo_labels
                            kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                            removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                            labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                            labels[removed_pseudo_labels] = 255
                        else:
                            raise ValueError(f"Unknown type of pseudo_ablation={self.pseudo_ablation}")
                    elif self.pseudo_soft == "soft_uncertain":
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]

                    if self.classif_adaptive_factor:
                        # Number of old/bg pixels that are certain
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                        # Number of old/bg pixels
                        den =  mask_background.float().sum(dim=(1,2))
                        # If all old/bg pixels are certain the factor is 1 (loss not changed)
                        # Else the factor is < 1, i.e. the loss is reduced to avoid
                        # giving too much importance to new pixels
                        classif_adaptive_factor = num / den
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if self.classif_adaptive_min_factor:
                            classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.classif_adaptive_min_factor)

            optim.zero_grad()
            outputs, features = model(images, ret_intermediate=self.ret_intermediate)

            # xxx BCE / Cross Entropy Loss
            if self.pseudo_soft is not None:
                loss = soft_crossentropy(
                    outputs,
                    labels,
                    outputs_old,
                    mask_valid_pseudo,
                    mask_background,
                    self.pseudo_soft,
                    pseudo_soft_factor=self.pseudo_soft_factor
                )
            elif not self.icarl_only_dist:
                if self.ce_on_pseudo and self.step > 0:
                    assert self.pseudo_labeling is not None
                    assert self.pseudo_labeling == "entropy"
                    # Apply UNCE on:
                    #   - all new classes (foreground)
                    #   - old classes (background) that were not selected for pseudo
                    loss_not_pseudo = criterion(
                        outputs,
                        original_labels,
                        mask=mask_background & mask_valid_pseudo  # what to ignore
                    )

                    # Apply CE on:
                    # - old classes that were selected for pseudo
                    _labels = original_labels.clone()
                    _labels[~(mask_background & mask_valid_pseudo)] = 255
                    _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &
                                                                                 mask_valid_pseudo]
                    loss_pseudo = F.cross_entropy(
                        outputs, _labels, ignore_index=255, reduction="none"
                    )
                    # Each loss complete the others as they are pixel-exclusive
                    loss = loss_pseudo + loss_not_pseudo
                elif self.ce_on_new:
                    _labels = labels.clone()
                    _labels[_labels == 0] = 255
                    loss = criterion(outputs, _labels)  # B x H x W
                else:
                    loss = criterion(outputs, labels)  # B x H x W
                    # print(outputs, labels)
            else:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

            if self.sample_weights_new is not None:
                sample_weights = torch.ones_like(original_labels).to(device, dtype=torch.float32)
                sample_weights[original_labels >= 0] = self.sample_weights_new
            
            if sample_weights is not None:
                loss = loss * sample_weights
            loss = classif_adaptive_factor * loss
            loss = loss.mean()  # scalar

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(
                    outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                )

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features['body'], features_old['body'])

            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                if self.lkd_mask is not None and self.lkd_mask == "oldbackground":
                    kd_mask = labels < self.old_classes
                elif self.lkd_mask is not None and self.lkd_mask == "new":
                    kd_mask = labels >= self.old_classes
                else:
                    kd_mask = None

                if self.temperature_apply is not None:
                    temp_mask = torch.ones_like(labels).to(outputs.device).to(outputs.dtype)

                    if self.temperature_apply == "all":
                        temp_mask = temp_mask / self.temperature
                    elif self.temperature_apply == "old":
                        mask_bg = labels < self.old_classes
                        temp_mask[mask_bg] = temp_mask[mask_bg] / self.temperature
                    elif self.temperature_apply == "new":
                        mask_fg = labels >= self.old_classes
                        temp_mask[mask_fg] = temp_mask[mask_fg] / self.temperature
                    temp_mask = temp_mask[:, None]
                else:
                    temp_mask = 1.0

                if self.kd_need_labels:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, labels, mask=kd_mask
                    )
                else:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, mask=kd_mask
                    )

                if self.kd_new:  # WTF?
                    mask_bg = labels == 0
                    lkd = lkd[mask_bg]

                if kd_mask is not None and self.kd_mask_adaptative_factor:
                    lkd = lkd.mean(dim=(1, 2)) * kd_mask.float().mean(dim=(1, 2))
                lkd = torch.mean(lkd)

            if self.pod is not None and self.step > 0:
                attentions_old = features_old["attentions"]
                attentions_new = features["attentions"]

                if self.pod_logits:
                    attentions_old.append(features_old["sem_logits_small"])
                    attentions_new.append(features["sem_logits_small"])
                elif self.pod_large_logits:
                    attentions_old.append(outputs_old)
                    attentions_new.append(outputs)

                pod_loss = features_distillation8(
                   attentions_old,
                   attentions_new,
                   index_new_class=self.old_classes,
                   nb_current_classes=self.nb_current_classes,
                   nb_new_classes=self.nb_new_classes
                ) + features_distillation_channel(
                  attentions_old,
                  attentions_new,
                  index_new_class=self.old_classes,
                  nb_current_classes=self.nb_current_classes,
                  nb_new_classes=self.nb_new_classes)


            if self.entropy_min > 0. and self.step > 0:
                mask_new = labels > 0
                entropies = entropy(torch.softmax(outputs, dim=1))
                entropies[mask_new] = 0.
                pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()

            if self.kd_scheduling:
                lkd = lkd * math.sqrt(self.nb_current_classes / self.nb_new_classes)

            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            loss_tot = loss + lkd + lde + l_icarl + pod_loss + loss_entmin

            with amp.scale_loss(loss_tot, optim) as scaled_loss:
                scaled_loss.backward()

            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    with amp.scale_loss(l_reg, optim) as scaled_loss:
                        scaled_loss.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + pod_loss.item(
            ) + loss_entmin.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}"
                )
                logger.info(
                    f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, POD {pod_loss} EntMin {loss_entmin}"
                )
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def find_median(self, train_loader, device, logger, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        if self.pseudo_nb_bins is not None:
            nb_bins = self.pseudo_nb_bins

        histograms = torch.zeros(self.nb_current_classes, nb_bins).float().to(self.device)

        for cur_step, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=False)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device).float(),
                accumulate=True
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )  

        logger.info("Approximating median")
        for c in range(self.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        if self.step_threshold is not None:
            self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None, end_task=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        model.module.in_eval = True
        if self.model_old is not None:
            self.model_old.in_eval = True

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        if self.step > 0 and self.align_weight_frequency == "epoch":
            model.module.align_weight(self.align_weight)
        elif self.step > 0 and self.align_weight_frequency == "task" and end_task:
            model.module.align_weight(self.align_weight)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels, name) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (
                    self.lde_flag or self.lkd_flag or self.icarl_dist_flag
                ) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)


                # # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag and not self.kd_need_labels:
                    lkd = self.lkd_loss(outputs, outputs_old).mean()

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(), labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(
                    f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                )

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])


def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)



def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))




