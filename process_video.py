import os.path
import cv2
import lane_detection as ld
import vehicle_detection as vd
from camera_calibration import calibrate_camera
from train_classifier import get_trained_classifier
from line import Line
from moviepy.editor import VideoFileClip

left_line = Line()
right_line = Line()

lines = {
    'left': left_line, 
    'right': right_line
}

calibration = []

def pipeline(img):
    global lines, calibration

    undistorted_img = ld.undistort(img, calibration[0], calibration[1])
    size_for_warp = (int(undistorted_img.shape[1]/2), undistorted_img.shape[0])
    M, M_inv = ld.get_warp_matrix(undistorted_img)
    warped = cv2.warpPerspective(undistorted_img, M, size_for_warp, flags=cv2.INTER_LINEAR)
    
    thresholded_image = ld.threshold_image(warped)

    left = ld.get_line_fit(thresholded_image, lines, 'left')
    right = ld.get_line_fit(thresholded_image, lines, 'right')
    
    if ld.get_left_right_compliance(left, right) and left.detected:
        left.update()

    if ld.get_left_right_compliance(right, left) and right.detected:  
        right.update()

    overlayer = ld.get_overlayer(warped, lines)
    size_for_unwarp = (int(undistorted_img.shape[1]), undistorted_img.shape[0])
    unwarped = cv2.warpPerspective(overlayer, M_inv, size_for_unwarp, flags=cv2.INTER_LINEAR)
    vehicles_detected = vd.detect_vehicles(undistorted_img, svc, X_scaler)
    return ld.overlay_image(vehicles_detected, unwarped, lines)


filename = input('Enter a name of video to process (including filetype): ')
if os.path.isfile(filename): 
  
    print('calibrating camera...')
    calibration = calibrate_camera()

    print('Training classifier...')
    svc, X_scaler = get_trained_classifier()



    project_output = 'output-' + filename
    clip = VideoFileClip(filename)
    test_clip = clip.fl_image(pipeline)
    test_clip.write_videofile(project_output, audio=False)

    print('processed video printed as:' + project_output)

else:
  print('file not found.')



