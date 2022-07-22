#этот файл я редактировал из муртазы1,он работает ,но изображение слишком быстро мигает-нужно доделать
import cv2
import numpy as np
cap = cv2.VideoCapture(0)#добавил я
w_path ="weights/yolov3-tiny.weights"
cfg_p="cfg/yolov3-tiny.cfg"

whT = 320
confThreshold =0.5
nmsThreshold= 0.2
# Load Yolo
net = cv2.dnn.readNet(w_path, cfg_p)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
#img = cv2.imread(0)#я закоментировал


# Detecting objects


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
#for out in outs:
    #for detection in out:
        #scores = detection[5:]
        #class_id = np.argmax(scores)
        #confidence = scores[class_id]
        #if confidence > 0.5:
            # Object detected
            #center_x = int(detection[0] * width)
            #center_y = int(detection[1] * height)
            #w = int(detection[2] * width)
            #h = int(detection[3] * height)

            # Rectangle coordinates
            #x = int(center_x - w / 2)
            #y = int(center_y - h / 2)

            #boxes.append([x, y, w, h])
            #confidences.append(float(confidence))
            #class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

def findObjects(outs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        #i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        with open(classesFile,'rt') as f:
              classNames=[line.rstrip() for line in f]
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
     sucess,img = cap.read()#добавил я
     img = cv2.resize(img, None, fx=1, fy=1)
     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

     net.setInput(blob)
     outs = net.forward(output_layers)
     findObjects(outs,img)

     height, width, channels = img.shape
     cv2.imshow("Image", img)
     cv2.waitKey(0.5)
     cv2.destroyAllWindows()

