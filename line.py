import numpy as np

class Line():
    def __init__(self):
        self.recent_amount = 7
        self.weights = [.2,.2,.2,.1,.1,.1,.1]
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.recent_fits = []
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.allx = None
        self.ally = None

    def preliminary_update(self, line_fit, fitx):
        if line_fit is not None:
            
            if self.should_update(fitx):
                self.detected = True
                self.current_fit = line_fit
                self.allx = fitx
                if self.best_fit is None:
                    self.update()
            else:
                self.detected = False
    
    def should_update(self, new_fitx):
        should_update = True
        
        if  self.bestx is not None:
            
            if new_fitx[719] > self.bestx[719] + 50 or new_fitx[719] < self.bestx[719] - 50:
                should_update = False

        return should_update
    
    def update(self):
        self.detected = True
        self.recent_fits.append(self.current_fit)
        self.recent_fits = self.recent_fits[-self.recent_amount:]
        self.best_fit = np.average(self.recent_fits, axis=0, weights=self.weights[:len(self.recent_fits)]).astype(int)
        self.recent_xfitted.append(self.allx)
        self.recent_xfitted = self.recent_xfitted[-self.recent_amount:]
        self.bestx = np.average(self.recent_xfitted, axis=0, weights=self.weights[:len(self.recent_xfitted)]).astype(int)