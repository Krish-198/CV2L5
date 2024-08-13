import cv2,numpy as np
img=cv2.imread("eye.png")
color=(0,229,255)
thickness=3
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b=cv2.blur(gray,(7,7))
c=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=32,maxRadius=40)
if c is not None:
    c=np.uint16(np.around(c))
    for i in c[0,:]:
        a,b,r=i[0],i[1],i[2]
        cv2.circle(img,(a,b),r,color,thickness)
        cv2.circle(img,(a,b),1,color,thickness)
        cv2.imshow("Object Recognition",img)
        cv2.waitKey(0)



img2=cv2.imread("blob.png",0)
cv2.imshow("Object Recognition",img2)
p=cv2.SimpleBlobDetector_Params()
p.filterByArea=True
p.minArea=100
cv2.waitKey(0)
cv2.destroyAllWindows()