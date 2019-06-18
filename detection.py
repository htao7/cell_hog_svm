from cell_hog_svm_func import *

hog = Load_svm("Ellie.yml")
os.chdir("./detection/")
cv2.namedWindow('img',cv2.WINDOW_NORMAL)

for imgfile in os.listdir('.'):
    if imgfile.endswith('.tif'):
        img = cv2.imread(imgfile,0)
        img = cv2.GaussianBlur(img,(3,3),0)
        r,c = img.shape
        img_detected, detected_box_list = Detect(img,hog)
        # labelfile = open(imgfile + '.txt','w')
        # for i,box in enumerate(detected_box_list):
        #     labelfile.write("%i %i %f %f %f %f\n" % (i,box.label,box.x,box.y,box.w,box.h))
        # labelfile.close()
        # cv2.imshow('img',img_detected)
        # cv2.resizeWindow('img',(int(c/(r/1000)),1000))
        # cv2.waitKey(0)
        cv2.imwrite('l' + imgfile,img_detected)
cv2.destroyAllWindows()
