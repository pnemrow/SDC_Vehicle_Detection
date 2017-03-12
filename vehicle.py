import numpy as np

class Vehicle():
    def __init__(self):
        self.detected = False
        self.n_total_detections = 0
        self.n_consec_detections = 0
        self.n_consec_nondetections = 0
        self.current_bbox = None
        self.recent_bbox = []
        self.best_bbox = None
        
    def update(self, bbox):
            self.detected = True
            self.current_bbox = bbox
            self.recent_bbox.append(bbox)
            self.recent_bbox = self.recent_bbox[-50:]
            self.best_bbox = np.mean(self.recent_bbox, axis=0).astype(int)
            
    def clear_detections(self):
        if self.detected == True:
            self.detected = False
            self.n_total_detections += 1
            self.n_consec_detections += 1
            self.n_consec_nondetections = 0
        else:
            self.detected = False
            self.n_consec_nondetections += 1
            self.n_consec_detections = 0