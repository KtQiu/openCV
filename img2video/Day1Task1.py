import cv2
import numpy as np


def img2Video():
    # s = input("Please input the size: ")
    #
    # slist = s.split()
    # size = (int(slist[0]), int(slist[1]))
    # if size == (0, 0):
    #     size = (551, 768)
    fps = int(input("please input the fps:"))
    wid = int(input("please input the width:"))
    hei = int(input("please input the height:"))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fps = 1
    out = cv2.VideoWriter('out.avi', fourcc, fps, (wid,hei))
    for i in range(17):
        if i >= 10:
            fileName = "stitching\img{0:2d}.JPG".format(i)
        else:
            fileName = "stitching\img0{0:1d}.JPG".format(i)
        print(fileName)
        img = cv2.imread(fileName, 1)
        for j in range(fps):
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Deng & Qiu"
            img = cv2.putText(img, text, (70, 300), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
            img = cv2.resize(img, (wid,hei), interpolation=cv2.INTER_CUBIC)
            out.write(img)


    # font = cv2.FONT_HERSHEY_SIMPLEX
    # img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # text = "Deng & Qiu"
    # img = cv2.putText(img, text, (70, 300), font, 2, (0, 255, 255), 2, cv2.LINE_AA)

    out.release()
    cv2.destroyAllWindows()

img2Video()

