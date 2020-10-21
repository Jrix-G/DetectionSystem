import cv2

cascadeClassifierPath = "haarcascade_fullbody.xml"
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

cascadeClassifierPath2 = "haarcascade_frontalface_alt.xml"
cascadeClassifier2 = cv2.CascadeClassifier(cascadeClassifierPath2)

profileface = "haarcascade_profileface.xml"
cascadeClassifier3 = cv2.CascadeClassifier(profileface)

eye = "haarcascade_eye.xml"
cascadeClassifier4 = cv2.CascadeClassifier(eye)

lowerbody = "haarcascade_lowerbody.xml"
cascadeClassifier6 = cv2.CascadeClassifier(lowerbody)

cap = cv2.VideoCapture(0)


while(cap.isOpened()):
    _, frame = cap.read()

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detectedBody = cascadeClassifier.detectMultiScale(grayImage, scaleFactor= 1.3, minNeighbors=10)
    detectedFaces = cascadeClassifier2.detectMultiScale(grayImage, scaleFactor= 1.3, minNeighbors=10)
    detectedProfile = cascadeClassifier3.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=10)
    detectedEye = cascadeClassifier4.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=10)
    detectedLowerBody = cascadeClassifier6.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=10)

    for (x,y,width, height) in detectedFaces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (100, 255, 0), 3)
        print("Je détecte une tête")
        
        for (x,y,width, height) in detectedEye:
            cv2.rectangle(frame, (x,y), (x+width, y+height), (50, 255, 50), 1)
            print("Je détecte des yeux")
    
    for (x,y,width, height) in detectedProfile:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 255, 0), 1)
        print("Je détecte un profil")

    for (x,y,width, height) in detectedBody:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (0, 255, 0), 2)
        print("Je détecte un corps") 

    for (x,y,width, height) in detectedLowerBody:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (0, 255, 0), 3)
        print("Je détecte un lower corps")        

    cv2.imshow("resultat", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
