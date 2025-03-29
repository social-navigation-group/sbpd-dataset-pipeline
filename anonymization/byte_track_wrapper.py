import torch
import os
import os.path as osp
import cv2
from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model
from yolox.tracker.byte_tracker import BYTETracker
from yolox.data.data_augment import preproc

class ByteTrackWrapper:
    def __init__(self, model_path, exp_file, device="cuda", conf=0.5, nms=0.7, input_size=640, fp16=False):
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        exp_name = os.path.splitext(os.path.basename(exp_file))[0] 
        self.exp = get_exp(exp_file, exp_name)
        self.exp.test_conf = conf
        self.exp.nmsthre = nms
        self.exp.test_size = (input_size, input_size)
        self.fp16 = fp16

        self.model = self.exp.get_model().to(self.device)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if fp16:
            self.model = self.model.half()
        self.tracker = BYTETracker(
            args=type("Args", (), {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "aspect_ratio_thresh": 1.6,
                "min_box_area": 10,
                "mot20": False
            }),
            frame_rate=30
        )

    def update(self, frame):
        orig_h, orig_w = frame.shape[:2]
        img, ratio = preproc(frame, self.exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)

        tlwhs, ids = [], []
        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [orig_h, orig_w], self.exp.test_size)
            for t in online_targets:
                tlwh = t.tlwh
                if tlwh[2] * tlwh[3] > 10 and (tlwh[2] / tlwh[3]) < 1.6:
                    tlwhs.append(tlwh)
                    ids.append(t.track_id)

        return tlwhs, ids
