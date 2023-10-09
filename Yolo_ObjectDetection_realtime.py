import cv2
import numpy as np
#Load Yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names", "rt") as f:
    classes = f.read().rstrip('\n').split('\n')
    print(classes)
layer_name = net.getLayerNames()
output_layer = [layer_name[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#Loading image

img = cv2.VideoCapture(0)
#address = "http://192.168.195.30:4747/video" for wificamera droidcam installed in android phone
#img.open(address)


#Detecting Object
while True:
    _, frame =img.read()
    height, width, channel = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416),(0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer)
#Showing information on screen
    class_ids = []
    confidense = []
    bboxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #cv2.circle(img, (center_x, center_y),10,(0,255,0),2)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                #cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
                bboxes.append([x,y,w,h])
                confidense.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(bboxes,confidense,0.5,0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(bboxes)):
        if i in indexes:
            x,y,w,h = bboxes[i]
            lable = str(classes[class_ids[i]])
            lcolor = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y),(x+w,y+h),lcolor,2)
            cv2.putText(frame, lable, (x,y+30),font,3,lcolor,3)

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key == 27:
        break
img.release()
cv2.destroyAllWindows()