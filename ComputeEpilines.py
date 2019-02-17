# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID: 5292195
"""
"""
reference taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
and
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
"""

import cv2
import numpy as np

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    cv2.imwrite(name,img)
    #print("\n****" + name + " written to disk****")
    #print("height:",len(img))
    #print("width:",len(img[0]))
    
"""
function to draw lines on an image based on the epipolar line equations recieved 
from the function.
"""
def drawEpipolarLines(targetImg,targetlines,targetpts,linecolors):
    timg = []
    for coeff,pt,color in zip(targetlines,targetpts,linecolors):
        x1 = 0 
        y1 = int(-(coeff[2]/coeff[1]))
        x2 = targetImg.shape[1]
        y2 = int(-(((coeff[0]/coeff[1])*x2) + (coeff[2]/coeff[1])))
        timg = cv2.line(targetImg, (x1,y1), (x2,y2), tuple(color),1)
        timg = cv2.circle(timg,tuple(pt), 3, tuple(color), -1)
    return timg
    
def computeEpilines(colorimage,srcpts,dstpts,imageNumber,FundamentalMatrix,linecolors):
    sucubaeplines = cv2.computeCorrespondEpilines(srcpts,imageNumber,FundamentalMatrix)
    sucubaeplines = sucubaeplines.reshape(-1,3)
    sucubalineImg = drawEpipolarLines(colorimage,sucubaeplines,dstpts,linecolors)
    return sucubalineImg   
    
matchCount = 4
        
sucuba1 = cv2.imread("tsucuba_left.png",0)
sucuba2 = cv2.imread("tsucuba_right.png",0)
sucuba1color = cv2.imread("tsucuba_left.png",1)
sucuba2color = cv2.imread("tsucuba_right.png",1)

sc1color = sucuba1color.copy()
sc2color = sucuba2color.copy()

sift = cv2.xfeatures2d.SIFT_create()

keypt_1, descp_1 = sift.detectAndCompute(sucuba1,None)
keypt_2, descp_2 = sift.detectAndCompute(sucuba2,None)

scout1=cv2.drawKeypoints(sc1color,keypt_1,sc1color)  #drawKeypoints(sourceimage,keypoints,outputimage)
scout2=cv2.drawKeypoints(sc2color,keypt_2,sc2color)

writeImage("task2_sift1.jpg",scout1)
writeImage("task2_sift2.jpg",scout2)

bf_matcher = cv2.BFMatcher()
calcmatches = bf_matcher.knnMatch(descp_1,descp_2, k=2)
validMs = []
sourcePts = []
destPts = []
for m,n in calcmatches:
    if m.distance < 0.75*n.distance:
        validMs.append(m)
        sourcePts.append(keypt_1[m.queryIdx].pt)
        destPts.append(keypt_2[m.trainIdx].pt)

sourcePts = np.int32(sourcePts)
destPts = np.int32(destPts)

matchedImgs = cv2.drawMatches(sucuba1color,keypt_1,sucuba2color,keypt_2,validMs,None,flags=2)
writeImage("task2_matches_knn.jpg",matchedImgs)

FundamentalMatrix, mask = cv2.findFundamentalMat(sourcePts,destPts,cv2.RANSAC,20)
mask = np.ndarray.flatten(mask).tolist()
print("\nFundamental Matrix: \n",FundamentalMatrix)

"""
10 random inliers are calculated from the keypoints
"""
fmask = np.zeros((len(mask)))
inlierIndxs = np.where(np.array(mask) == 1)[0]
np.random.shuffle(inlierIndxs)
inlierIndxs = inlierIndxs[:10]
fmask[inlierIndxs.tolist()] = 1
srcInlierPts = []
destInlierPts = []
npvalidMs = np.array(validMs)
truncGoodPts = npvalidMs[inlierIndxs.tolist()]

for m in truncGoodPts:
    srcInlierPts.append(keypt_1[m.queryIdx].pt)
    destInlierPts.append(keypt_2[m.trainIdx].pt)
    
srcInlierPts = np.int32(srcInlierPts)
destInlierPts = np.int32(destInlierPts)
linecolors = [[0  ,255,  0],
              [255,0  ,0  ],
              [255,255,255],
              [0  ,0  ,255],
              [255,255,0  ],
              [238,130,238],
              [255,128,  0],
              [0, 255, 255],
              [255,228,181],
              [152,251,152]]

np.random.shuffle(linecolors)           

"""
the epilines are calculated for the corresponding points for both the images
"""
scubarightimg = computeEpilines(sucuba2color,srcInlierPts,destInlierPts,1,FundamentalMatrix,linecolors)
writeImage("task2_epi_right.jpg",scubarightimg)

scubaleftimg = computeEpilines(sucuba1color,destInlierPts,srcInlierPts,2,FundamentalMatrix,linecolors)
writeImage("task2_epi_left.jpg",scubaleftimg)

"""
Here the disparity map is calculated using the stereo function
since the image is shifted to right, we take the image starting from 63
It is divided by a contant to normalize 
"""
stobject = cv2.StereoBM_create()
disparity = stobject.compute(sucuba1,sucuba2)
print("\n\nDisparity: \n",disparity)
disparity = disparity[:,63:]/3.5
writeImage("task2_disparity.jpg",disparity)

