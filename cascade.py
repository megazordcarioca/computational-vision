import cv2 as cv

source_cascade = 'cars.xml'
source_cars = 'f1.mp4'

capture = cv.VideoCapture(source_cars)
car_cascade = cv.CascadeClassifier(source_cascade)

while True:
        ret, img = capture.read()
        if(type(img) == None):
            break
        grayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(grayScale)

        # Draw rectangles in cars

        for (x,y,w,h) in cars:
            cv.rectangle(img,(x,y),(x+w, y+h),(0,155,255),1)
        cv.imshow('F1 Cars - AV2 Using Haar Cascade', img)
        if cv.waitKey(33) == 27:
            break

cv.destroyAllWindows()