"""
YOLACT (ResNet-50-FPN) builder - the 1-Stage prototype-mask reference. Seg task.

Paper: Bolya, Zhou, Xiao & Lee, "YOLACT: Real-time Instance Segmentation" (ICCV 2019, arXiv 1904.02689).
Code : github.com/dbolya/yolact (yolact.py, layers/modules/multibox_loss.py, layers/box_utils.py, data/config.py
       yolact_resnet50_config), reimplemented here; backbone / FPN are the framework's ResNet-50 + FPN
       (detectors/retinanet, torchvision / detectron2 classes).

    backbone   ResNet-50 (C3-C5) + FPN P3-P7 (P6 / P7 by stride-2 conv from P5, no ReLU: yolact FPN with
               use_conv_downsample, relu_downsample_layers False)
    protonet   on P3: 3 x (conv 3x3 256 + ReLU) -> bilinear 2x -> conv 3x3 256 + ReLU -> conv 1x1 32 + ReLU
               -> 32 prototypes at stride 4 (152 x 152 for 608)
    head       shared over levels: conv 3x3 256 + ReLU, then box (4A), class (C+1)A, coefficient (32A, tanh); A = 3
    anchors    one scale per level x ratios (1, 1/2, 2); yolact scales 24/48/96/192/384 at 550 px, scaled to 608
    matching   IoU >= 0.5 positive, < 0.4 negative, best prior of every gt forced positive; SSD box encoding
               (variances 0.1 / 0.2)
    loss       box smooth L1 x 1.5 + softmax CE with OHEM (neg:pos 3) x 1 + mask BCE x 6.125 + semantic BCE x 1
               mask BCE at prototype resolution, cropped with the gt box, divided by the gt box area in prototype
               pixels (mask_proto_normalize_emulate_roi_pooling), <= 100 positives per image (masks_to_train);
               box, class and mask terms are divided by the number of positives; semantic aux head = 1x1 conv on P3
    inference  class probs = softmax without background; mask = sigmoid(coefficients . prototypes) cropped with the box

Raw contract: [B, R, 5 + C] (cx cy w h, objectness 1, softmax class probs); aux = coefficients + prototypes;
decode_masks kind "full" (tasks/seg_task.py upsamples, crops with the box, thresholds 0.5).

[Unified] no ImageNet weights, trainable BN + PreciseBN (BN ResNet-50, micro batch 2: same as RetinaNet), common
init, no mean / std, 608 input, Adam 1e-4, flip + YOLO-paper augmentation (instead of SSD photometric / expand /
crop), common post-processing and standard NMS (instead of Fast NMS), FPN outputs without the yolact ReLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from detectors.anchor_free.common import HeadStats, to_raw
from detectors.retinanet.fpn import build_retinanet_fpn
from detectors.retinanet.resnet import build_resnet
from detectors.yolo11_seg.build import crop_mask, split_seg_targets

NM = 32
STRIDES = (8, 16, 32, 64, 128)
SCALES_550 = (24, 48, 96, 192, 384)
RATIOS = (1.0, 0.5, 2.0)
VARIANCES = (0.1, 0.2)
POS_THR, NEG_THR, NEGPOS, MASKS_TO_TRAIN = 0.5, 0.4, 3, 100
BOX_ALPHA, CONF_ALPHA, MASK_ALPHA, SEM_ALPHA = 1.5, 1.0, 6.125, 1.0


def make_priors(feat_sizes, img_size, device):
    """[R, 4] cx cy w h (px); per cell the 3 aspect ratios, yolact make_priors order (cell-major)."""
    out = []
    for (h, w), stride, s550 in zip(feat_sizes, STRIDES, SCALES_550):
        scale = s550 * img_size / 550.0
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cx = (xs.float() + 0.5) / w * img_size
        cy = (ys.float() + 0.5) / h * img_size
        per = []
        for ar in RATIOS:
            r = ar ** 0.5
            per.append(torch.stack([cx, cy, torch.full_like(cx, scale * r), torch.full_like(cx, scale / r)], -1))
        out.append(torch.stack(per, 2).reshape(-1, 4))
    return torch.cat(out, 0)


def cxcywh_to_xyxy(b):
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)


def encode(gt_xyxy, priors):
    c = (gt_xyxy[:, :2] + gt_xyxy[:, 2:]) / 2
    wh = (gt_xyxy[:, 2:] - gt_xyxy[:, :2]).clamp(min=1e-6)
    return torch.cat([(c - priors[:, :2]) / (VARIANCES[0] * priors[:, 2:]),
                      torch.log(wh / priors[:, 2:]) / VARIANCES[1]], 1)


def decode(loc, priors):
    c = priors[..., :2] + loc[..., :2] * VARIANCES[0] * priors[..., 2:]
    wh = priors[..., 2:] * torch.exp(loc[..., 2:] * VARIANCES[1])
    return torch.cat([c - wh / 2, c + wh / 2], -1)


class ProtoNet(nn.Module):
    def __init__(self, c=256, nm=NM):
        super().__init__()
        self.pre = nn.Sequential(*[m for _ in range(3) for m in (nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True))])
        self.post = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, nm, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, p3):
        x = self.pre(p3)
        return self.post(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class YOLACT(nn.Module):
    def __init__(self, num_classes, c=256):
        super().__init__()
        self.num_classes = num_classes
        A = len(RATIOS)
        self.backbone = build_resnet(50)
        self.fpn = build_retinanet_fpn(self.backbone.out_channels, c, top_block_in="p5", relu_before_p7=False)
        self.proto = ProtoNet(c)
        self.upfeature = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(c, 4 * A, 3, padding=1)
        self.conf_layer = nn.Conv2d(c, (num_classes + 1) * A, 3, padding=1)
        self.mask_layer = nn.Conv2d(c, NM * A, 3, padding=1)
        self.semantic = nn.Conv2d(c, num_classes, 1)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _heads(self, x):
        feats = self.fpn(self.backbone(x))
        B = x.shape[0]
        locs, confs, coefs = [], [], []
        for f in feats:
            u = self.upfeature(f)
            locs.append(self.bbox_layer(u).permute(0, 2, 3, 1).reshape(B, -1, 4))
            confs.append(self.conf_layer(u).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes + 1))
            coefs.append(torch.tanh(self.mask_layer(u)).permute(0, 2, 3, 1).reshape(B, -1, NM))
        priors = make_priors([f.shape[-2:] for f in feats], x.shape[2], x.device)
        return feats, torch.cat(locs, 1), torch.cat(confs, 1), torch.cat(coefs, 1), self.proto(feats[0]), priors

    def forward(self, x, targets=None):
        feats, loc, conf, coef, proto, priors = self._heads(x)
        boxes = decode(loc, priors[None])
        raw = to_raw(boxes, F.softmax(conf, -1)[..., 1:])
        if targets is None:
            if self.training:                                    # PreciseBN forward
                return raw
            return raw, {"coef": coef, "proto": proto}
        B, S = x.shape[0], x.shape[2]
        tgt = split_seg_targets(targets, B, S)
        loss_dict, pos, fg_labels = self.loss(feats[0], loc, conf, coef, proto, priors, tgt, S)
        flat = conf.detach()[..., 1:].reshape(-1, self.num_classes)
        self.stats.update(loss_dict, F.softmax(conf.detach(), -1)[..., 1:].reshape(-1, self.num_classes).max(-1).values,
                          flat, pos.view(-1), fg_labels)
        return loss_dict["losses"], raw

    def loss(self, p3, loc, conf, coef, proto, priors, tgt, img_size):
        B, R, _ = loc.shape
        prior_xyxy = cxcywh_to_xyxy(priors)
        conf_t = torch.zeros(B, R, dtype=torch.long, device=loc.device)
        loc_t = torch.zeros_like(loc)
        idx_t = torch.zeros(B, R, dtype=torch.long, device=loc.device)
        for i, t in enumerate(tgt):
            if len(t["labels"]) == 0:
                continue
            ov = box_iou(t["boxes"], prior_xyxy)                    # [G, R]
            best_truth_ov, best_truth = ov.max(0)
            best_prior = ov.max(1).indices
            best_truth_ov[best_prior] = 2.0                         # yolact match(): force each gt's best prior
            best_truth[best_prior] = torch.arange(len(best_prior), device=loc.device)
            lab = t["labels"][best_truth] + 1
            lab[best_truth_ov < POS_THR] = -1                       # neutral
            lab[best_truth_ov < NEG_THR] = 0                        # background
            conf_t[i], idx_t[i] = lab, best_truth
            loc_t[i] = encode(t["boxes"][best_truth], priors)
        pos = conf_t > 0
        num_pos = pos.sum().clamp(min=1).float()

        loss_b = F.smooth_l1_loss(loc[pos], loc_t[pos], reduction="sum") * BOX_ALPHA

        # OHEM (yolact ohem_conf_loss): hardest negatives by background loss, neg:pos = 3
        with torch.no_grad():
            bg = torch.logsumexp(conf, -1) - conf[..., 0]
            bg[pos | (conf_t < 0)] = 0
            rank = bg.argsort(1, descending=True).argsort(1)
            n_neg = (NEGPOS * pos.sum(1, keepdim=True)).clamp(max=R - 1)
            neg = (rank < n_neg) & ~pos & (conf_t == 0)
        sel = pos | neg
        loss_c = F.cross_entropy(conf[sel], conf_t[sel], reduction="sum") * CONF_ALPHA

        _, nm, mh, mw = proto.shape
        loss_m = coef.sum() * 0.0
        loss_s = coef.sum() * 0.0
        sem_logits = self.semantic(p3)                               # [B, C, H/8, W/8]
        sh, sw = sem_logits.shape[-2:]
        for i, t in enumerate(tgt):
            if len(t["labels"]):
                m = t["masks"][None].float()
                seg_t = torch.zeros(self.num_classes, sh, sw, device=loc.device)
                ds = (F.interpolate(m, (sh, sw), mode="bilinear", align_corners=False)[0] > 0.5).float()
                for k in range(len(t["labels"])):                    # yolact semantic_segmentation_loss
                    seg_t[t["labels"][k]] = torch.max(seg_t[t["labels"][k]], ds[k])
                loss_s = loss_s + F.binary_cross_entropy_with_logits(sem_logits[i], seg_t, reduction="sum")
            p = torch.nonzero(pos[i]).squeeze(1)
            if p.numel() == 0:
                continue
            if p.numel() > MASKS_TO_TRAIN:
                p = p[torch.randperm(p.numel(), device=p.device)[:MASKS_TO_TRAIN]]
            gi = idx_t[i, p]
            gt_m = (F.interpolate(t["masks"][None].float(), (mh, mw), mode="bilinear", align_corners=False)[0]
                    > 0.5).float()[gi]
            pred = (coef[i, p] @ proto[i].view(nm, -1)).view(-1, mh, mw)
            box = t["boxes"][gi] * torch.tensor([mw / img_size, mh / img_size] * 2, device=loc.device)
            bce = crop_mask(F.binary_cross_entropy_with_logits(pred, gt_m, reduction="none"), box)
            area = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])).clamp(min=1.0)
            loss_m = loss_m + (bce.sum(dim=(1, 2)) / area).sum()
        loss_m = loss_m * MASK_ALPHA
        loss_s = loss_s / (sh * sw) * SEM_ALPHA
        losses = (loss_b + loss_c + loss_m) / num_pos + loss_s
        fg_labels = conf_t[pos] - 1
        return ({"loss_box": loss_b / num_pos, "loss_cls": loss_c / num_pos, "loss_mask": loss_m / num_pos,
                 "loss_sem": loss_s, "losses": losses}, pos, fg_labels)

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        proto = aux["proto"][b]
        nm, mh, mw = proto.shape
        return {"kind": "full", "prob": (aux["coef"][b, rows] @ proto.view(nm, -1)).sigmoid().view(-1, mh, mw)}

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLACT has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return YOLACT(num_classes)
