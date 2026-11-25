"""torchvision.ops placeholders/implementations used by onnx2torch imports."""
import torch


def nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.int64)


def box_convert(boxes, in_fmt, out_fmt):
    if in_fmt == out_fmt:
        return boxes
    x, y, w, h = boxes.unbind(-1)
    if in_fmt == "xyxy" and out_fmt == "xywh":
        return torch.stack((x, y, w - x, h - y), -1)
    if in_fmt == "xywh" and out_fmt == "xyxy":
        return torch.stack((x, y, x + w, y + h), -1)
    if in_fmt == "xyxy" and out_fmt == "cxcywh":
        return torch.stack(((x + w) / 2, (y + h) / 2, w - x, h - y), -1)
    if in_fmt == "cxcywh" and out_fmt == "xyxy":
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)
    if in_fmt == "xywh" and out_fmt == "cxcywh":
        return torch.stack((x + w / 2, y + h / 2, w, h), -1)
    if in_fmt == "cxcywh" and out_fmt == "xywh":
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, w, h), -1)
    raise NotImplementedError(f"box_convert: {in_fmt} -> {out_fmt}")


def roi_align(*_args, **_kwargs):
    raise NotImplementedError("roi_align nincs megvalósítva a shimben")


def batched_nms(*_args, **_kwargs):
    raise NotImplementedError("batched_nms nincs megvalósítva a shimben")