import torch
import numpy as np
from ultralytics import YOLO

#====================================================
class SegmentationTeacher:
    
    #====================================================
    def __init__(self):
        self.model = YOLO("yolo11l.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #====================================================
    def forward(self, image_path: str, temperature: float = 1.0) -> dict:
        background = 0
        results = self.model(image_path) # inference
        result = results[0]
        num_classes = len(result.names) # classes in output

        orig_h, orig_w = result.orig_shape
        
        # hard mask output
        hard_label_mask = np.full((orig_h, orig_w), background, dtype=np.int32)
        
        # soft mask output
        soft_label_mask = np.zeros((orig_h, orig_w, num_classes), dtype=np.float32)
        soft_label_mask[:, :, background] = 1.0
        
        class_ids = []
        confidences = []
        
        # go through each detected object
        if len(result.boxes) > 0:

            # bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy() 

            # classes
            cls_ids = result.boxes.cls.cpu().numpy().astype(int) 

            # confidences
            confs = result.boxes.conf.cpu().numpy()
            
            for i, (bbox, cls_id, conf) in enumerate(zip(boxes, cls_ids, confs)):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # clip bounding box to image
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # mask wih label
                hard_label_mask[y1:y2, x1:x2] = cls_id
                
                # converting the confidence to a probability
                prob = np.exp(conf / temperature)
                    
                # this splits the probability between the class and the background.
                # this is because whatever probability was not part of the object 
                # is assumed to be part of the background
                soft_label_mask[y1:y2, x1:x2, cls_id] = prob
                soft_label_mask[y1:y2, x1:x2, background] = 1.0 - prob
                
                class_ids.append(int(cls_id))
                confidences.append(float(conf))
        
        # normalize probabilities
        prob_sum = soft_label_mask.sum(axis=2, keepdims=True)
        prob_sum[prob_sum == 0] = 1.0
        soft_label_mask = soft_label_mask / prob_sum
        
        return {
            'hard_mask': hard_label_mask,
            'soft_probs': soft_label_mask,
            'class_ids': class_ids,
            'confidences': confidences,
            'original_shape': (orig_h, orig_w),
            'num_classes': num_classes
        }
    
    #====================================================
    def to_tensor(self, labels_dict: dict, device: str) -> dict:
        tensor_dict = {}
        
        # hard label to tensor
        tensor_dict['hard_mask'] = torch.from_numpy(labels_dict['hard_mask']).long().to(device)
        
        # soft label to tensor
        if labels_dict['soft_probs'] is not None:
            soft_probs = labels_dict['soft_probs'].transpose(2, 0, 1) # C, H, W
            tensor_dict['soft_probs'] = torch.from_numpy(soft_probs).float().to(device)
        
        tensor_dict['class_ids'] = labels_dict['class_ids']
        tensor_dict['confidences'] = labels_dict['confidences']
        tensor_dict['original_shape'] = labels_dict['original_shape']
        tensor_dict['num_classes'] = labels_dict['num_classes']
        return tensor_dict