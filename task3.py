import cv2
import os
from ultralytics import YOLO
from collections import Counter



model = YOLO('yolov8n.pt')

#1
results = model('imageS/')
print(model.names)
for result in results:
    prediction = result.boxes.cls.numpy()
    real = os.path.basename(result.path).split('_')[1:]
    real[-1] = real[-1].split('.')[0]
    real = list(map(int, real))
    intersection = Counter(real) & Counter(prediction)
    accuracy = int((sum(intersection.values())/len(real) * 100))
    print(f'{os.path.basename(result.path)} Точность предсказания: {accuracy}%')

#Изучив model (обратившись к names), будем называть изображения набором индексов,
# которое соответсвуют тому, что есть на
# В задании требуется 20 изображений, на каждом изображении несколько объектов, все используемые в изображениях, перечислены ниже
#22: 'zebra', 23: 'giraffe',
#49: 'orange', 47: 'apple', 46: 'banana',
#43: 'knife', 44: 'spoon', 42: 'fork'
#results = model('22_22_23.jpg')

# results = model(r'C:\ifds\i2.jpg')
# results = model(r'C:\ifds\i3.jpg')
# results = model(r'C:\ifds\i4.jpg')
# results = model(r'C:\ifds\i5.jpg')

#2

#3
# Инициализация веб-камеры
# Инициализация веб-камеры
# Инициализация веб-камеры

video_cap = cv2.VideoCapture()
#если нужно видео: video_cap = cv2.VideoCapture("niazAAA.mp4")

confidence_threshold = 0.7
ret, frame = video_cap.read()
detections = model(frame)[0]

while True:
    # Считывание кадра
    ret, frame = video_cap.read()
    detections = model(frame)[0]

    for data in detections.boxes.data.tolist():
        xmin = int(data[0])
        xmax = int(data[2])
        ymin = int(data[1])
        ymax = int(data[3])
        confidence = data[4]

        class_name = detections.names[int(data[5])]
        print(detections.names)
        exit(1)
        if confidence < confidence_threshold:
            continue
        cv2.rectangle(frame, (xmin, ymin), (xmax,ymax), (0,255,0), 2)
        cv2.putText(frame,class_name , (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0,255,0))


    cv2.imshow('OOOOOO', frame)
    cv2.waitKey(2)




    #
    # # Выполнение вывода на кадре
    # results = model(frame)
    #
    # # Отображение кадра с обнаруженными объектами
    # for result in results:
    #     for img in result.render()[0]:
    #         cv2.imshow('YOLO Object Detection', img)

    # # Проверка нажатия клавиши 'q' для выхода из цикла
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Освобождение захвата
video_cap.release()
cv2.destroyAllWindows()

#4 захват с видеофайла


