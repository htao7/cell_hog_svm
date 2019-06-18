from cell_hog_svm_func import *

os.chdir("./training/")
for i in range(int(len(os.listdir('.'))/2)):
    img = cv2.imread(str(i) + '.tif')
    img_h,img_w,_ = img.shape
    labelfile = open(str(i) + '.txt', 'r')
    for line in labelfile:
        if line == "E" or line == "E\n":
            break
        word = line.split(" ")
        label = int(word[0])
        x = float(word[1]) * img_w
        y = float(word[2]) * img_h
        w = float(word[3]) * 1.1 * img_w
        h = float(word[4]) * 1.1 * img_h
        x = int(x - w / 2)
        y = int(y - h / 2)
        w = int(w)
        h = int(h)
        if label == 1:
            cv2.rectangle(img,(x,y),(x + w,y + h),(0, 0, 255),2)
        elif label == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',img)
    cv2.resizeWindow('img',(int(img_w/(img_h/1000)),1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
