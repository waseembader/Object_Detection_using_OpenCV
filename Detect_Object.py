import numpy as np
import cv2

MIN_MATCH_COUNT=70

detector=cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

# input the cropped image inside Training folder for training
trainImg=cv2.imread('Training/train11.png',1)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

# Image where the object is to be detected
QueryImgBGR=cv2.imread('unmarked/1.jpg',1)

QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
matches=flann.knnMatch(queryDesc,trainDesc,k=2)

goodMatch=[]
for m,n in matches:
    if(m.distance<0.75*n.distance):
            goodMatch.append(m)
if(len(goodMatch)>MIN_MATCH_COUNT):
    tp=[]
    qp=[]
    for m in goodMatch:
        tp.append(trainKP[m.trainIdx].pt)
        qp.append(queryKP[m.queryIdx].pt)
    tp,qp=np.float32((tp,qp))
    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
    h,w=trainImg.shape[:-1]
    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    queryBorder=cv2.perspectiveTransform(trainBorder,H)
    cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,0,255),5)
    cv2.imwrite('output/result.jpg',QueryImgBGR)
    print('Object detected, marked and saved the marked image in output folder')

else:
    print ('Object not found')

