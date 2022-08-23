import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0) 
desenho = mp.solutions.drawing_utils 
estilo = mp.solutions.drawing_styles 
mao = mp.solutions.hands 

with mao.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while webcam.isOpened():
        verificador, frame = webcam.read() 

        if not verificador:
            break

        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        frame.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        pontos =[]

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                desenho.draw_landmarks(
                    frame,
                    handLandmarks,
                    mao.HAND_CONNECTIONS
                )
       
                for id, lm in enumerate(handLandmarks.landmark):
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    #Adiciona os numeros dos pontos
                    #cv2.putText(frame, str(id),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
                    pontos.append((cx,cy))

            #pontos mais altos dos dedos, sem dedao
            dedos = [8,12,16,20]
            contador = 0
            like = 0

            #Ponto mais alto do dedao tem q ficar a esquerda para ser considerado como contagem
            if pontos[4][0] < pontos[2][0]:
                    contador += 1
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]:
                    contador += 1
            
            if pontos[4][1] < pontos[2][0]:
                like += 1
            for x in dedos:
                if pontos[x][0] > pontos[x-2][0]:
                    like += 1
            
            if like == 5:
                cv2.putText(frame, "Deixa o Like!",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,128,0), 5)
            else:
                cv2.putText(frame, str(contador),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,128,0), 5)

        cv2.imshow('Reconhecimento de maos', frame)

        if cv2.waitKey(5) == 27: #tecla 27 corresponde ao esc
            break

webcam.release()
cv2.destroyAllWindows()