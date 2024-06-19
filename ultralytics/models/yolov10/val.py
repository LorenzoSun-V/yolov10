from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            # 如果conf是默认值0.001，那么就返回所有结果
            if self.args.conf == 0.001:
                return preds
            else:
                mask = preds[..., 4] > self.args.conf
                if self.args.classes is not None:
                    mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
                return [p[mask[idx]] for idx, p in enumerate(preds)]
        else:
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            if self.args.conf == 0.001:
                return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
            else:
                # 如果conf是默认值0.001，那么就返回所有结果
                preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
                mask = preds[..., 4] > self.args.conf
                if self.args.classes is not None:
                    mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)

                return [p[mask[idx]] for idx, p in enumerate(preds)]