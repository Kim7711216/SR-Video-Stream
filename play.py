import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter.fourcc('x', '2', '6', '4')
out = cv2.VideoWriter('output.mp4', fourcc, 20, (640, 480))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:        
        #out.write(frame)
        print(frame)
        cv2.imshow('video', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()