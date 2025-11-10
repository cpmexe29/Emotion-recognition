import os
import warnings
import logging

# Отключаем ВСЕ предупреждения и логи
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Оптимизация использования GPU

# Подавляем все предупреждения Python
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Отключаем логирование TensorFlow до импорта
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

import cv2
from deepface import DeepFace

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось подключить камеру")
    exit()

# Переменные для управления анализом
last_analysis_time = 0
analysis_interval = 1  # Анализировать каждую секунду
current_emotion = "Определение..."
current_confidence = 0

print("Запуск распознавания эмоций. Нажмите 'q' для выхода...")

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()

    if not ret:
        print("Ошибка захвата видео")
        break

    # Получение текущего времени для интервального анализа
    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    # Анализ эмоций с заданным интервалом (чтобы не нагружать систему)
    if current_time - last_analysis_time > analysis_interval:
        try:
            # Анализ доминирующей эмоции с помощью DeepFace
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,  # Продолжать даже если лицо не обнаружено
                silent=True
            )

            # Обработка результатов анализа
            if results and isinstance(results, list):
                analysis = results[0]
                emotion = analysis['dominant_emotion']  # Основная эмоция
                confidence = analysis['emotion'][emotion]  # Уверенность в %
                current_emotion = emotion
                current_confidence = confidence
                print(f"Эмоция: {emotion} ({confidence:.1f}%)")
            else:
                current_emotion = "Лицо не обнаружено"
                current_confidence = 0

        except Exception as e:
            print(f"Ошибка анализа: {e}")
            current_emotion = "Ошибка"
            current_confidence = 0

        last_analysis_time = current_time

    # Отображение результата поверх видео
    cv2.putText(frame, f"Emotion: {current_emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {current_confidence:.1f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # Показ кадра в окне
    cv2.imshow('Распознавание эмоций', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
