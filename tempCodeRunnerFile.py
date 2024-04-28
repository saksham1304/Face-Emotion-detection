
    cap = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)