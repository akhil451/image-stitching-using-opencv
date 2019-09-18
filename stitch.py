import cv2
import numpy as np
import os

input_folder='images'
output_folder='images_output'
def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# def read_images(folder):
#     images=[cv2.imread(os.path.join(os.getcwd(),folder,image)) for image in os.listdir(folder)]
#     return images

left= cv2.imread(input_folder+'/sedona_left_01.png')
right= cv2.imread(input_folder+'/sedona_right_01.png')
def crop(result):
    #reference -- pyimagesearch.com
    result = cv2.copyMakeBorder(result, 10, 10, 10, 10,
    cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    c=cnts[0]
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()


    while cv2.countNonZero(sub) > 0:

        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)


    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    c=cnts[0]
    (x, y, w, h) = cv2.boundingRect(c)


    result= result[y:y + h, x:x + w]
    return result

    
    
def stitch(left,right,output_folder,use_opencv):
    create_folder(output_folder)
    descriptor = cv2.xfeatures2d.SIFT_create()

    print(left.shape)
    print(right.shape)
    if not use_opencv:
        ratio=0.75;min_match=10
        print('tests')
        gray_left = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
        gray_right=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
        (kps_left, features_left) = descriptor.detectAndCompute(gray_left, None)
        (kps_right, features_right) = descriptor.detectAndCompute(gray_right, None)
        # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(features_left,features_right)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(features_left, features_right, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        if len(good_points) > min_match:
            image1_kp = np.float32(
                [kps_left[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kps_right[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)

        result = cv2.warpPerspective(right, H,(right.shape[1] + left.shape[1], right.shape[0]))
        result[0:left.shape[0], 0:left.shape[1]] = left 
        cv2.imwrite(os.path.join(output_folder,'uncropped.png'),result)
#         plt.imshow(result)
        cropped=crop(result)
        cv2.imwrite(os.path.join(output_folder,'cropped.png'),cropped)
#         plt.imshow(cropped)

    else:
        stitcher=cv2.createStitcher()
        (status, stitched) = stitcher.stitch([left,right])
        if status==0:
            cv2.imwrite(os.path.join(output_folder,'opencv_stitcher_uncropped.png'),stitched)
            cropped=crop(stitched)
            cv2.imwrite(os.path.join(output_folder,'opencv_stitcher_cropped.png'),cropped)
                
stitch(left,right,output_folder,True)
                
            
        
        
        
            