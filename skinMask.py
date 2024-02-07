#importing packages
from imutils import paths
import cv2
import numpy as np

#loading all dataset path in imagePaths Variable
lis =['e','ee','u','oo','i','au','k','kh','g','gh','ch','cha','j','jh','t','th','d','dh','n','ta','tha','da','dha','na','p','f','b','bh','ma','ya','ra','la','va','sha','s','ha','lla','jah','jaha']
for j in lis:
    imagePaths = list(paths.list_images("C:/Users/Dev Prajapati/Desktop/Sign Language  Files/MarathiDatasetv1/"+j))
    path = len(imagePaths)
    print(imagePaths)


    for i in range(path):
        imagePath = imagePaths[i]
        image = cv2.imread(imagePath)
        #cv2.imshow('Input image', image)
        # converting BGR to HSV
        HSV_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #cv2.imshow('HSV IMAGE', HSV_img)

        # setting hsv values based on skin color
        # lower_color = np.array([108, 23, 82])
        # upper_color = np.array([179, 255, 255])

        lower_color = np.array([5, 50, 50], dtype="uint8")
        upper_color = np.array([15, 255, 255], dtype="uint8")
        mask = cv2.inRange(HSV_img, lower_color, upper_color)
        #cv2.imshow('Mask 1', mask)

        # morphological operations,erosion followed by dilation
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        #cv2.imshow('opening process', dilation)

        # gaussian blur
        blur = cv2.GaussianBlur(dilation, (5, 5), 0)
        #cv2.imshow('Gausssian Blur', blur)

        # masking with original image
        mask2 = cv2.bitwise_and(image, image, mask=blur)

        # convert from color to grayscale
        #gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('C:/Users/Dev Prajapati/Desktop/CNN/final_marathi_mask_dataset/'+j+"/"+str(i)+'.jpg', mask2)
        #cv2.imshow('Final Image', gray)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        #cv2.waitKey(0)
cv2.destroyAllWindows()