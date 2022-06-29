import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "\yolact")

from data import COLORS, cfg, set_cfg
from yolact import Yolact

from utils import timer
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
import cv2

from Autodrivegame.object import detectedObject

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
color_cache = defaultdict(lambda: {})

class objectDetector():
    def __init__(self, trained_model, 
                cuda=True, display_lincomb=False, score_threshold=0, crop=False, display_text=True, display_bboxes=True,
                mask_proto_debug=False, fast_nms=True, cross_class_nms=False,
                top_k=0, display_masks=True, display_scores=True, display_pos=True, tracker=False):

        self.display_lincomb = display_lincomb
        self.score_threshold = score_threshold
        self.crop = crop
        self.display_text = display_text
        self.display_bboxes = display_bboxes
        self.display_scores = display_scores
        self.display_masks = display_masks
        self.display_pos = display_pos
        self.mask_proto_debug = mask_proto_debug
        self.fast_nms = fast_nms
        self.cross_class_nms = cross_class_nms

        self.top_k = top_k

        self.model_path = SavePath.from_str(trained_model)
        self.config = self.model_path.model_name + '_config'
        set_cfg(self.config)
        
        if tracker: self.tracker = tracker

        with torch.no_grad():
            if cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')

            print('Loading model...', end='')
            self.net = Yolact()
            self.net.load_weights(trained_model)
            self.net.eval()
            print(' Done.')

            self.net = self.net.cuda()

            self.net.detect.use_fast_nms = self.fast_nms
            self.net.detect.use_cross_class_nms = self.cross_class_nms
            
            cfg.mask_proto_debug = self.mask_proto_debug

    def prep_display(self, dets_out, img):
        detected_object_list = []
        h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = self.display_lincomb,
                                            crop_masks        = self.crop,
                                            score_threshold   = self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k]
            if cfg.eval_mask_branch:
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            score = scores[j]
               
            _object = detectedObject(cfg.dataset.class_names[classes[j]], classes[j], [x1, y1, x2, y2], round(score, 3), masks[j].byte().cpu().numpy())

            detected_object_list.append(_object)

        return detected_object_list


    def detectObject(self, image):
        with torch.no_grad():
            frame = torch.from_numpy(image).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

            detected_object_list = self.prep_display(preds, frame)

        if self.tracker:
            detected_object_list = self.tracker.track(detected_object_list)

        return detected_object_list

