import cv2
import numpy as np
import os
from random import random

class Box:
    def __init__(self,label,x,y,w,h):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def Training(model):
    label_list = []
    hog_list = []
    for i in range(int(len(os.listdir('./training/'))/2)):
        label_list_i, hog_list_i = Calc_hog(i)
        label_list.extend(label_list_i)
        hog_list.extend(hog_list_i)
    label_list = np.array(label_list)
    hog_list = np.array(hog_list)
    Train_svm(label_list, hog_list,model)
    return

def Read_label(i):
    img = cv2.imread('./training/' + str(i) + '.tif',0)
    img = cv2.GaussianBlur(img,(3,3),0)
    labelfile = open('./training/' + str(i) + '.txt','r')
    box_list = []
    for line in labelfile:
        #print(line)
        if line == "E" or line == "E\n":
            break
        word = line.split(" ")
        label = int(word[0])
        x = float(word[1])
        y = float(word[2])
        w = float(word[3]) * 1.1
        h = float(word[4]) * 1.1
        box_list.append(Box(label,x,y,w,h))
    labelfile.close()
    return (img,box_list)

def Calc_hog(i):
    Create_neglabel(i, 20)
    img, box_list = Read_label(i)
    img_h,img_w = img.shape
    label_list = []
    hog_list = []
    hog = cv2.HOGDescriptor("hog.xml")
    for box in box_list:
        l = max(box.w * img_w,box.h * img_h)
        if box.y * img_h - l / 2 < 0 or box.y * img_h + l / 2 > img_h or box.x * img_w - l / 2 < 0 or box.x * img_w + l / 2 > img_w:
            continue
        img_box = np.copy(img[int(box.y * img_h - l / 2):int(box.y * img_h + l / 2),\
                              int(box.x * img_w - l / 2):int(box.x * img_w + l / 2)])
        if abs(img_box.shape[0] - img_box.shape[1]) > 2 or img_box.shape[0] == 0:
            continue
        img_box_resized = cv2.resize(img_box,(64,64),interpolation=cv2.INTER_AREA)
        #cv2.imshow('box', img_box_resized)
        #cv2.waitKey(0)
        if box.label == 0:
            enhance1 = cv2.flip(img_box_resized, 0)
            enhance2 = cv2.flip(img_box_resized, 1)
            enhance3 = cv2.flip(img_box_resized, -1)
            hist1 = hog.compute(enhance1)
            hist2 = hog.compute(enhance2)
            hist3 = hog.compute(enhance3)
            label_list.append(-1)
            hog_list.append(hist1)
            label_list.append(-1)
            hog_list.append(hist2)
            label_list.append(-1)
            hog_list.append(hist3)
            box.label = -1

        if box.label == 1:
            enhance1 = cv2.flip(img_box_resized,0)
            enhance2 = cv2.flip(img_box_resized,1)
            enhance3 = cv2.flip(img_box_resized,-1)
            hist1 = hog.compute(enhance1)
            hist2 = hog.compute(enhance2)
            hist3 = hog.compute(enhance3)
            label_list.append(1)
            hog_list.append(hist1)
            label_list.append(1)
            hog_list.append(hist2)
            label_list.append(1)
            hog_list.append(hist3)

        hist = hog.compute(img_box_resized)
        label_list.append(box.label)
        hog_list.append(hist)

    label_list = np.array(label_list)
    hog_list = np.array(hog_list)
    return (label_list,hog_list)


def Create_neglabel(i,num):
    labelfile = open('./training/' + str(i) + '.txt', 'r')
    lastline = labelfile.readlines()[-1]
    labelfile.close()
    #print (i,lastline)
    if lastline == "E" or lastline == "E\n":
        return
    labelfile = open('./training/' + str(i) + '.txt', 'a')
    for j in range(num):
        x = random()
        y = random()
        w = random() * min(2 * x,2 * (1 - x)) * 0.5
        h = random() * min(2 * y,2 * (1 - y)) * 0.5
        labelfile.write("-1 %f %f %f %f\n" % (x,y,w,h))
    labelfile.write('E')
    labelfile.close()
    return

def Train_svm(label_list,hog_list,model):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_EPS_SVR)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(10)
    svm.setP(0.1)
    svm.setNu(0.5)
    svm.train(hog_list,cv2.ml.ROW_SAMPLE,label_list)
    svm.save(model)
    return

def Load_svm(model):
    svm = cv2.ml.SVM_load(model)
    hog = cv2.HOGDescriptor("hog.xml")
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.append(sv,-rho)
    hog.setSVMDetector(sv)
    return (hog)

def Detect(img,hog):
    (boundingbox, weight) = hog.detectMultiScale(img,winStride=(4,4),padding=(4,4),scale=1.1,finalThreshold=0)
    #print (max(weight),min(weight))
    boundingbox = boundingbox.tolist()
    weight = weight.ravel().tolist()
    nms_index = cv2.dnn.NMSBoxes(boundingbox,weight,1,0.1)
    detected_box_list = []
    img_h,img_w = img.shape
    #print (nms_index)
    if len(nms_index) != 0:
        for i,index in enumerate(nms_index.flatten()):
            (x,y,w,h) = boundingbox[index]
            cv2.rectangle(img,(x,y),(x + w,y + h),(0, 0, 255),2)
            cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2, cv2.LINE_AA)
            detected_box_list.append(Box(0,(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w / 1.1, h / img_h / 1.1))
    return (img,detected_box_list)


if __name__ == '__main__':
    Training("Ellie.yml")
    #print (label_list,hog_list)
    img2 = cv2.imread('1.tif',0)
    img2 = cv2.GaussianBlur(img2,(5,5),0)
    hog = Load_svm("Ellie.yml")
    img_detected,detected_box_list = Detect(img2,hog)
    r,c = img_detected.shape
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.imshow('img',img_detected)
    cv2.resizeWindow('img',(int(c/(r/1000)),1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
