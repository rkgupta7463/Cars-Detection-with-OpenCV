import cv2 as cv
import numpy as np

cap=cv.VideoCapture("D:/Computer Vision Practices-YOLO/Car Detection/resourcs/los_angeles.mp4")
cap.set(3,600)
cap.set(4,400)


## Initialize Substructor
Tracker_algo = cv.createBackgroundSubtractorMOG2()

count_line_position=550
## min width of rectange and max
min_width_rect=80 
min_height_rect=80 


def center_point(x,y,w,h):
    x1=int(w//2)
    y1=int(h//2)
    cx=x+x1
    cy=y+y1

    return cx,cy

detect=[]

offset=6 ## offset allowable error between pixel

carCounter=0

while True:
    s,img=cap.read()


    if img is not None:
        imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        imgBlur=cv.GaussianBlur(imgGray,(3,3),2)
 
        # applying our tracking algorothms on each image frame
        img_sub=Tracker_algo.apply(imgBlur)
        dilate=cv.dilate(img_sub,np.ones((5,5)))
        kernal=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        dilateada=cv.morphologyEx(dilate,cv.MORPH_CLOSE,kernal)
        dilateada=cv.morphologyEx(dilateada,cv.MORPH_CLOSE,kernal)
        counter,h=cv.findContours(dilateada,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cv.line(img,(0,count_line_position),(1280,count_line_position),(0,0,0),2)

        for (i,c) in enumerate(counter):
             (x,y,w,h)=cv.boundingRect(c,)
             validate_counter=(w>=min_width_rect) and (h>=min_height_rect)
             if not validate_counter:
                 continue
             cv.rectangle(img,(x,y),(x+w,y+h),(0,150,200),2)
             cv.putText(img,f"Vehicle counter {carCounter}",(x,y-20),cv.FONT_HERSHEY_COMPLEX,1,(0,60,200),2)

             center_point_counter=center_point(x,y,w,h) 
             detect.append(center_point_counter)
             cv.circle(img,center_point_counter,4,(0,0,250),-1)

             for (x,y) in detect:
                 if y<(count_line_position+offset) and y>(count_line_position-offset):
                     carCounter+=1
                 cv.line(img,(0,count_line_position),(1280,count_line_position),(0,127,20),3)                
                 detect.remove((x,y))                   
                 print("Car counter:- ",carCounter)   
             print(type(carCounter))
        
        cv.putText(img,f"Vehicle counter : {carCounter}",(450,70),cv.FONT_HERSHEY_COMPLEX,2,(0,60,200),2)



        # cv.imshow("Detector Video",dilateada)
        cv.imshow("Cars Video",img)
    else:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()