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
p=cv2.SimpleBlobDetector_Params()
p.filterByArea=True
p.minArea=100
p.filterByCircularity=True
p.minCircularity=0.9
p.filterByConvexity=True
p.minConvexity=0.2
p.filterByInertia=True
p.minInertiaRatio=0.01
t=cv2.SimpleBlobDetector_create(p)
r=t.detect(img2)
e=np.zeros((7,7))
v=cv2.drawKeypoints(img2,r,e,color,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
noblobs=len(r)
text="No blobs detected- " + str(noblobs)
cv2.putText(v,text,(20,500),cv2.FONT_HERSHEY_SIMPLEX,1,color,thickness)

cv2.imshow("Object Recognition",v)

cv2.waitKey(0)
cv2.destroyAllWindows()
