#实际的原理是用摄像头一帧一阵的读取图片，
#然后以图片的形式去不断的show出来
#假如想写的话，就是一帧一帧的以图片的形式写入

import cv2

cap = cv2.VideoCapture(0)     #0表示的是摄像头的编号，0表示笔记本内置，1或者其他的是外置的
while(1):
    ret, frame = cap.read()
    cv2.imshow("capture",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

