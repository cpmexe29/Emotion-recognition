import os
import warnings
import logging

# Отключаем лишние предупреждения и логи для чистого вывода
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Импортируем библиотеки после настройки окружения
import tensorflow as tf
import cv2
from deepface import DeepFace
import numpy as np

# Удаление лишних сообщений в консоль
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

# Подключаем камеру
cap = cv2.VideoCapture(0)

# Грузим детектор лиц из OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Не могу подключиться к камере")
    exit()

# Настройки для анализа
last_analysis_time = 0
analysis_interval = 1  # Анализируем раз в секунду
current_emotion = "Определение..."
current_confidence = 0
face_detected = False
consecutive_no_face_frames = 0

print("Запускаю распознавание эмоций. Нажми 'q' чтобы выйти...")


def put_text_ru(img, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """
    Выводим текст на кадр, если русские буквы не работают - заменяем на английские
    """
    # Вывод текста в окно
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)


# Главный цикл обработки видео
while True:
    # Читаем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        print("Не могу получить кадр с камеры")
        break

    # Переводим в черно белое для детектора лиц, лучше распознаёт эмоции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ищем лица в кадре
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Рисуем рамки вокруг найденных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Задержка распознования эмоции перед другой
    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    # Анализируем эмоции интервалами
    if current_time - last_analysis_time > analysis_interval:
        if len(faces) > 0:
            try:
                # Отправляем кадр в DeepFace для анализа эмоций
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv'
                )

                if results and isinstance(results, list):
                    analysis = results[0]  # Берем первое найденное лицо

                    # Проверяем, что DeepFace тоже увидел лицо
                    if (analysis.get('region', {}).get('w', 0) > 20 and
                            analysis.get('region', {}).get('h', 0) > 20):

                        emotion = analysis['dominant_emotion']
                        confidence = analysis['emotion'][emotion]

                        # Обновляем результаты
                        current_emotion = emotion
                        current_confidence = confidence
                        face_detected = True
                        consecutive_no_face_frames = 0

                        print(f"Распознано: {emotion} (уверенность: {confidence:.1f}%)")

                    else:
                        # DeepFace не распознал лицо или не может определить эмоцию
                        current_emotion = "No face"
                        current_confidence = 0
                        face_detected = False
                        consecutive_no_face_frames += 1
                        print("DeepFace не смог распознать лицо")

                else:
                    # Нет эмоции
                    current_emotion = "No face"
                    current_confidence = 0
                    face_detected = False
                    consecutive_no_face_frames += 1
                    print("Нет данных от анализатора")

            except Exception as e:
                # Если что-то пошло не так при анализе
                print(f"Ошибка при анализе: {e}")
                current_emotion = "Analysis error"
                current_confidence = 0
                face_detected = False
                consecutive_no_face_frames += 1
        else:
            # Лиц в кадре нет
            current_emotion = "No face"
            current_confidence = 0
            face_detected = False
            consecutive_no_face_frames += 1

            if consecutive_no_face_frames == 1:
                print("В кадре нет лиц")

        last_analysis_time = current_time

    # Показываем результаты на экране

    # Выбираем цвет текста: зеленый если лицо есть, красный если нет
    if face_detected:
        text_color = (0, 255, 0)  # Зеленый
        status_text = f"Emotion: {current_emotion}"
    else:
        text_color = (0, 0, 255)  # Красный
        status_text = "No face"

    # Выводим основную информацию
    put_text_ru(frame, status_text, (20, 40), 1, text_color, 2)

    # Дополнительная информация в зависимости от статуса
    if face_detected:
        confidence_text = f"Confidence: {current_confidence:.1f}%"
        put_text_ru(frame, confidence_text, (20, 80), 0.7, text_color, 2)
    else:
        # Подсказка пользователю
        info_text = "Point camera at face"
        put_text_ru(frame, info_text, (20, 80), 0.7, text_color, 2)

        # Показываем сколько лиц видит OpenCV
        faces_text = f"OpenCV faces: {len(faces)}"
        put_text_ru(frame, faces_text, (20, 110), 0.7, (255, 255, 255), 2)

    # Показываем обработанное видео
    cv2.imshow('Emotion Recognition', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем рабочие окна
cap.release()
cv2.destroyAllWindows()
print("Работа завершена.")
