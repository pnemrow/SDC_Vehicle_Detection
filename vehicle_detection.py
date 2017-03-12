import cv2
import numpy as np
from scipy.ndimage.measurements import label
import math
from vehicle import Vehicle
from train_classifier import *
import vehicle_detector_variables as vdv

recent_heat_maps = []
vehicle_list = []

def detect_vehicles(img, svc, X_scaler):
    y_windows = [(390, 510), (400,550),(460,800), (460,750)]
    scales = [1.5, 1.75,2, 3]
    out_img, heat_map = find_cars(img, y_windows, scales, svc, X_scaler)
    averaged_heat = average_heat_map(heat_map)
    labels = label(averaged_heat)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

def find_cars(img, y_windows, scales, svc, X_scaler):
    bboxes = []
    draw_img = np.copy(img)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    img = img.astype(np.float32)/255

    for scale, y_window in zip(scales, y_windows):
        
        ystart = y_window[0]
        ystop = y_window[1]
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch)
        
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        nxblocks = (ch1.shape[1] // vdv.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // vdv.pix_per_cell)-1 

        nfeat_per_block = vdv.orient*vdv.cell_per_block**2
        window = 64
        nblocks_per_window = (window // vdv.pix_per_cell)-1
        cells_per_step = 2
        nxsteps = ((nxblocks - nblocks_per_window) // cells_per_step)+2
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, vdv.orient, vdv.pix_per_cell, vdv.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, vdv.orient, vdv.pix_per_cell, vdv.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, vdv.orient, vdv.pix_per_cell, vdv.cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                right_window_xpos = xpos+nblocks_per_window

                if right_window_xpos > hog1.shape[1]:
                    xpos = xpos - (right_window_xpos - hog1.shape[1])
                    right_window_xpos = hog1.shape[1]
                    x_window = (right_window_xpos - xpos + 1) * vdv.pix_per_cell
                else: x_window = window

                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:right_window_xpos].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:right_window_xpos].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:right_window_xpos].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*vdv.pix_per_cell
                ytop = ypos*vdv.pix_per_cell
                x_right = xleft+x_window
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:x_right], (64,64))
                
                spatial_features = bin_spatial(subimg, size=vdv.spatial_size)
                hist_features = color_hist(subimg, nbins=vdv.hist_bins)
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) 
                test_prediction = svc.predict(test_features)
                conf = svc.decision_function(test_features)

                if test_prediction == 1 and conf > 0.4:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    x_win_draw = np.int(x_window*scale)

                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+x_win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    box = ((xbox_left, ytop_draw + ystart),(xbox_left+x_win_draw, ytop_draw + win_draw+ystart))
                    bboxes.append(box)

    heat = add_heat(heat,bboxes) 
    apply_threshold(heat, 0)
    heatmap = np.clip(heat, 0, 255)
    
    return draw_img, heatmap

def average_heat_map(current):
    global recent_heat_maps
    recent_heat_maps.append(current)

    if len(recent_heat_maps) > 10:
        recent_heat_maps.pop(0)
    
    totaled_heat_map = np.sum(recent_heat_maps, axis=0)
    thresholded = apply_threshold(totaled_heat_map, 25)
    return thresholded

        
def draw_labeled_bboxes(img, labels):
    global vehicle_list

    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        if len(vehicle_list) > 0:
            matched_v = [v for v in vehicle_list if determine_vehicle(v.best_bbox, bbox)]
            
            if len(matched_v) == 0:
                new_vehicle = Vehicle()
                new_vehicle.update(bbox)
                vehicle_list.append(new_vehicle)
            else:
                matched_v[0].update(bbox)
                
        else:
            new_vehicle = Vehicle()
            new_vehicle.update(bbox)
            vehicle_list.append(new_vehicle)
        
    for vehicle in vehicle_list:
        vehicle.clear_detections()
        if vehicle.n_consec_nondetections > 30:
            vehicle_list.remove(vehicle)
        elif vehicle.n_total_detections > 30:
            cv2.rectangle(img, tuple(vehicle.best_bbox[0]), tuple(vehicle.best_bbox[1]), (0,0,255), 6)
            
    return img

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def determine_vehicle(vehicle_bbox, new_bbox):
    margin = 100
    
    vehicle_midpoint = np.mean(vehicle_bbox, axis=0)
    new_midpoint = np.mean(new_bbox, axis=0)
    dist = math.hypot(vehicle_midpoint[0]-new_midpoint[0], vehicle_midpoint[1]-new_midpoint[1])
    
    return dist < margin