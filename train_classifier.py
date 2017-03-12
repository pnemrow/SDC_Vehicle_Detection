import os
import glob
import time
import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import vehicle_detector_variables as vdv

def get_trained_classifier():

    basedir = 'vehicles/'
    image_types = os.listdir(basedir)
    cars = []

    for imtype in image_types:
        cars.extend(glob.glob(basedir + imtype+'/*'))
        
    print('Number of vehicle images found:', len(cars))

    with open("cars.txt", 'w') as f:
        for fn in cars:
            f.write(fn +'\n')
            
    basedir = 'non-vehicles/'
    image_types = os.listdir(basedir)
    notcars = []

    for imtype in image_types:
        notcars.extend(glob.glob(basedir + imtype+'/*'))
        
    print('Number of Non-vehicle images found:', len(notcars))

    with open("notcars.txt", 'w') as f:
        for fn in notcars:
            f.write(fn +'\n')

    t=time.time()
    n_samples = 1000
    random_idxs = np.random.randint(0,len(cars), n_samples)
    test_cars = cars
    test_notcars = notcars

    car_features = extract_features(test_cars, color_space=vdv.color_space, 
                            spatial_size=vdv.spatial_size, hist_bins=vdv.hist_bins, 
                            orient=vdv.orient, pix_per_cell=vdv.pix_per_cell, 
                            cell_per_block=vdv.cell_per_block, 
                            hog_channel=vdv.hog_channel, spatial_feat=vdv.spatial_feat, 
                            hist_feat=vdv.hist_feat, hog_feat=vdv.hog_feat)
    notcar_features = extract_features(test_notcars, color_space=vdv.color_space, 
                            spatial_size=vdv.spatial_size, hist_bins=vdv.hist_bins, 
                            orient=vdv.orient, pix_per_cell=vdv.pix_per_cell, 
                            cell_per_block=vdv.cell_per_block, 
                            hog_channel=vdv.hog_channel, spatial_feat=vdv.spatial_feat, 
                            hist_feat=vdv.hist_feat, hog_feat=vdv.hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    X_scaler = StandardScaler().fit(X)

    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Using:',vdv.orient,'orientations',vdv.pix_per_cell,
        'pixels per cell and', vdv.cell_per_block,'cells per block and colorspace', vdv.color_space)
    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()

    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 8))

    return svc, X_scaler

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
                      pix_per_cell=8, cell_per_block=2, hog_channel=0,
                      spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []

    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)        

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=vdv.spatial_size)
            file_features.append(spatial_features)
            
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=vdv.hist_bins)
            file_features.append(hist_features)
            
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        vdv.orient, vdv.pix_per_cell, vdv.cell_per_block, 
                                        vis=False, feature_vec=True))
                
                hog_features = np.ravel(hog_features)   
            else:
                hog_features = get_hog_features(feature_image[:,:,vdv.hog_channel], vdv.orient, 
                                vdv.pix_per_cell, vdv.cell_per_block, vis=False, feature_vec=True)
            
            file_features.append(hog_features)
            
        features.append(np.concatenate(file_features))
    return features

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    if vis == True:
        features, hog_image = hog(img, orientations=vdv.orient, 
                                  pixels_per_cell=(vdv.pix_per_cell, vdv.pix_per_cell),
                                  cells_per_block=(vdv.cell_per_block, vdv.cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    else:      
        features = hog(img, orientations=vdv.orient, 
                       pixels_per_cell=(vdv.pix_per_cell, vdv.pix_per_cell),
                       cells_per_block=(vdv.cell_per_block, vdv.cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

