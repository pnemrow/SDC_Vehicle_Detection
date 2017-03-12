import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
from PIL import Image

global steering_wheel

def undistort(img, mtx, dist):
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_img

def get_warp_matrix(img):
    src = np.float32([
        [448., 479.],
        [832., 479.],
        [1472., 680.],
        [-192., 680.]
    ])

    dst = np.float32([
        [96., 0.],
        [544., 0.],
        [544., 720.],
        [96., 720.]
    ])

    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return M, M_inv

def threshold_image(img):
    kernel = np.ones((14,14),np.uint8)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    
    hls_l = hls[:,:,1]
    th_hls_l = cv2.morphologyEx(hls_l, cv2.MORPH_TOPHAT, kernel)
    hls_l_binary = np.zeros_like(th_hls_l)
    hls_l_binary[(th_hls_l > 20) & (th_hls_l <= 255)] = 1

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
    lab_l = lab[:,:,0]
    th_lab_l = cv2.morphologyEx(lab_l, cv2.MORPH_TOPHAT, kernel)
    lab_l_binary = np.zeros_like(th_lab_l)
    lab_l_binary[(th_lab_l > 20) & (th_lab_l <= 255)] = 1

    lab_b = lab[:,:,2]
    th_lab_b = cv2.morphologyEx(lab_b, cv2.MORPH_TOPHAT, kernel)
    lab_b_binary = np.zeros_like(th_lab_b)
    lab_b_binary[(th_lab_b > 5) & (th_lab_b <= 255)] = 1

    full_mask = np.zeros_like(th_hls_l)
    full_mask[(hls_l_binary == 1) | (lab_l_binary == 1) | (lab_b_binary == 1)] = 1

    kernel = np.ones((6,3),np.uint8)
    erosion = cv2.erode(full_mask,kernel,iterations = 1)

    return erosion

def get_line_fit(thresholded_image, lines, side):
    
    if lines[side].detected == True:
        lines[side] = line_in_windows(thresholded_image, lines[side])
    
    if lines[side].detected == False:
        lines = locate_line(thresholded_image, lines, side)
        
    return lines[side]

def get_left_right_compliance(target, compare):
    spacing_confirmed = confirm_spacing(target, compare)
    angle_confirmed = confirm_angle(target, compare)
    curve_confirmed = confirm_curve(target, compare)
    return spacing_confirmed and angle_confirmed and curve_confirmed

def get_overlayer(warped, lines):
    if lines is not None:
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([lines['left'].bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([lines['right'].bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        overlayer = cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    return overlayer

def overlay_image(original, overlayer, lines):
    global steering_wheel
    overlayed = cv2.addWeighted(original, 1, overlayer, 0.3, 0)

    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    img = np.zeros_like(original)

    ym_per_pix = 3/170
    xm_per_pix = 3.7/210
    leftx = lines['left'].bestx
    rightx = lines['right'].bestx

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    road_curve = left_curverad + right_curverad / 2
    road_curve = round(road_curve, 2)
    curve_string = "{:,}".format(road_curve)

    if road_curve > 3000:
        curve_string = 'Straight'
    else:
        curve_string = str(curve_string) + 'm'

    center_of_lines = (rightx[len(rightx)-1] + leftx[len(leftx)-1]) / 2
    distance_from_center = center_of_lines - img.shape[1] / 4
    distance_from_center = distance_from_center * xm_per_pix
    distance_from_center = round(distance_from_center, 2)
    center_offset = "{:,}".format(abs(distance_from_center))

    if distance_from_center > 0:
        center_offset = str(center_offset) + 'm left'
    else:
        center_offset = str(center_offset) + 'm right'

    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(overlayed,'Road Curve: ' + curve_string,(10,50), font, 1.4,(255,255,255),2)
    image = cv2.putText(image, 'Off Center: ' + center_offset,(10,100), font, 1.4,(255,255,255),2)

    if ((left_fit_cr[0] + right_fit_cr[0]) / 2) > 0:
        road_curve = -1 * road_curve
    
    background = Image.fromarray(image).convert("RGBA")
    background.paste(get_steer_wheel(road_curve), (540, 150), get_steer_wheel(road_curve))
    image = np.array(background)
    image = image[:,:,:3]
    
    return image

def locate_line(thresholded, lines, lane_side):
    histogram = np.sum(thresholded[thresholded.shape[0]/2:,:], axis=0)
    out_img = np.dstack((thresholded, thresholded, thresholded))*255
    midpoint = np.int(histogram.shape[0]/2)
   
    if lane_side == 'left':
        x_base = np.argmax(histogram[:midpoint])
    else:
        x_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    window_height = np.int(thresholded.shape[0]/nwindows)
    nonzero = thresholded.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    x_current = x_base
    margin = 30
    minpix = 50
    lane_inds = []

    for window in range(nwindows):
        win_y_low = thresholded.shape[0] - (window+1)*window_height
        win_y_high = thresholded.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2)
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        lane_inds.append(good_inds)
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
            
    lane_inds = np.concatenate(lane_inds)

    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    
    if x.shape[0] != 0:
        line_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, thresholded.shape[0]-1, thresholded.shape[0] )
        fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        radius = get_radius(fitx, ploty)
        lines[lane_side].preliminary_update(line_fit, fitx)
    
    return lines

def line_in_windows(thresholded, line):
    line_fit = fitx = radius = None
    fit = line.current_fit
    out_img = np.dstack((thresholded, thresholded, thresholded))*255
    nonzero = thresholded.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin))) 
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    
    if x.shape[0] != 0:
        line_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, thresholded.shape[0]-1, thresholded.shape[0] )
        fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        radius = get_radius(fitx, ploty)
        line.preliminary_update(line_fit, fitx)
    
    return line

def get_radius(line, ploty):
    y_eval = np.max(ploty)
    curverad = ((1 + (2*line[0]*y_eval + line[1])**2)**1.5) / np.absolute(2*line[0])
    ym_per_pix = 3/170
    xm_per_pix = 3.7/210
    fit_cr = np.polyfit(ploty*ym_per_pix, line*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    return curverad

def confirm_angle(left, right):
    ret_value = True

    if left.current_fit[1] > right.current_fit[1] + .6:
        ret_value = False
        
    if left.current_fit[1] < right.current_fit[1] -.6:
        ret_value = False
        
    return ret_value

def confirm_spacing(left, right):
    ret_value = True
    new_space = right.allx[719] - left.allx[719]
    average_space = right.bestx[719] - left.bestx[719]

    if new_space > average_space + 100 or new_space < average_space - 100:
        
        ret_value = False
        
    return ret_value

def confirm_curve(left, right):
    ret_value = True

    if left.current_fit[0] > right.current_fit[0] + .0007:#was .0005
        ret_value = False
        
    if left.current_fit[0] < right.current_fit[0] - .0007:# was .0005
        ret_value = False

    return ret_value

def get_steer_wheel(curve):
    global steering_wheel
    steering_wheel = Image.open('output_images/steering_wheel.png').convert("RGBA")
    steering_wheel= steering_wheel.resize((200,200),Image.ANTIALIAS)
    angle = (4 * (math.pi + math.asin(9/(curve*2)))) * 200.40
    return steering_wheel.rotate(angle)