import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math
from ultralytics import YOLO
from models.mtcnn import MTCNN
from .tools import round_all, sleep_ms
import cv2

def detect_face(img:np.ndarray|torch.Tensor, mtcnn:MTCNN, yolo:YOLO=None, yolo_margin:float=0.3, min_pixel:int=4000, min_conf:float=0.7) -> list:
    result = []

    if yolo is not None:
        assert len(img.shape) == 3
        pred = yolo.predict(img, verbose=False, stream=True)
        for i in pred:
            for j in i.boxes:
                if j.conf[0] < min_conf:
                    continue
                x1, y1, x2, y2 = j.xyxy[0]
                mx = x2-x1
                my = y2-y1
                x1, y1, x2, y2 = int(round_all(x1 - (mx*yolo_margin))), int(round_all(y1 - (my*yolo_margin))), int(round_all(x2 + (mx*yolo_margin))), int(round_all(y2 + (my*yolo_margin)))
                mtc_fr = img[y1:y2, x1:x2]
                if mtc_fr.shape[0] * mtc_fr.shape[1] > min_pixel:
                    mt_pred = mtcnn.detect(mtc_fr, landmarks=True)
                    if mt_pred[0] is not None:
                        sx1, sy1, sx2, sy2 = mt_pred[0][0]
                        sx1, sy1, sx2, sy2 = x1+int(round_all(sx1)), y1+int(round_all(sy1)), x1+int(round_all(sx2)), y1+int(round_all(sy2))
                        landmark = []
                        for ldmk in mt_pred[2][0]:
                            px, py = x1+int(ldmk[0]), y1+int(ldmk[1])
                            landmark.append([px,py])
                        result.append({"bbox": [sx1, sy1, sx2, sy2], "landmarks" : landmark})
    else:
        pass
        # assert isinstance(img, torch.Tensor)

        # pred = mtcnn.detect(img, landmarks=True)
        
        # for i in range(len(pred[0])):
        #     this_batch = []
        #     if pred[0][i] is not None:
        #         for j in range(len(pred[0][i])):
        #             x1, y1, x2, y2 = pred[0][i][j]
        #             x1, y1, x2, y2 = int(round_all(x1)), int(round_all(y1)), int(round_all(x2)), int(round_all(y2))

        #             landmark = []
        #             for ldmk in pred[2][i][j]:
        #                 px, py = int(round_all(ldmk[0])), int(round_all(ldmk[1]))
        #                 landmark.append([px, py])
        #             this_batch.append({"bbox": [x1, y1, x2, y2], "landmarks" : landmark})
        #         result.append(this_batch)
        #     else:
        #         result.append(None)

        # if pred[0] is not None:
        #     for i in range(len(pred[0])):
        #         x1, y1, x2, y2 = pred[0][i]
        #         x1, y1, x2, y2 = int(round_all(x1)), int(round_all(y1)), int(round_all(x2)), int(round_all(y2))

        #         landmark = []
        #         for ldmk in pred[2][i]:
        #             px, py = int(round_all(ldmk[0])), int(round_all(ldmk[1]))
        #             landmark.append([px, py])

        #         result.append({"bbox": [x1,y1,x2,y2], "landmarks": landmark})
    return result

def batching_detect_face(img:np.ndarray|torch.Tensor, mtcnn:MTCNN, min_pixel:int=4000, min_conf:float=0.7) -> list:
    result = []

    pred = mtcnn.detect(img, landmarks=True)

    if pred[0] is not None:
        for i in range(len(pred[0])):
            bbox = pred[0][i]
            conf = pred[1][i]
            landmk = pred[2][i]

            x1, y1, x2, y2 = list(map(lambda x: int(round(float(x))), bbox))

            if (x2-x1) * (y2-y1) < min_pixel:
                continue

            if conf < min_conf:
                continue

            landmark = []
            for ldmk in landmk:
                px, py = int(round(float(ldmk[0]))), int(round(float(ldmk[1])))
                landmark.append([px, py])

            result.append({"bbox": [x1, y1, x2, y2], "landmarks":landmark })

    return result

