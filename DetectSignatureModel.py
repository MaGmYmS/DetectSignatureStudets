import enum
import os
import random

import cv2
import numpy as np

from ultralytics import YOLO
from SelfDevelopment import delete_files_in_folder


class PredictClass(enum.Enum):
    FullName = "Full-name"
    Signature = "Signature"


class DetectSignatureModel:
    def __init__(self, number_train):
        my_best_model = f"runs/detect/train{number_train}/weights/best.pt"  # Загружаем модель
        model = YOLO(my_best_model)
        self.model = model

    def predict_detect_model(self, image):
        # получаем ширину и высоту картинки
        h, w, _ = image.shape

        # получаем предсказания по картинке
        results = self.model.predict(source=image, conf=0.50, verbose=False)

        # расшифровываем объект results
        bboxes_ = results[0].boxes.xyxy.tolist()
        bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
        confs_ = results[0].boxes.conf.tolist()
        confs = list(map(lambda x: int(x * 100), confs_))
        classes_ = results[0].boxes.cls.tolist()
        classes = list(map(lambda x: int(x), classes_))
        cls_dict = results[0].names
        class_names = list(map(lambda x: cls_dict[x], classes))

        # приводим дешифрированные данные в удобный вид
        result_array = []
        for index, val in enumerate(class_names):
            x_min, y_min, x_max, y_max = int(bboxes[index][0]), int(bboxes[index][1]), int(bboxes[index][2]), int(
                bboxes[index][3])
            result_array.append((x_min, y_min, x_max, y_max))
        return result_array, class_names

    def crop_and_save_image(self, image, clean_output_dir=False):
        output_dir = 'cropped_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if clean_output_dir:
            delete_files_in_folder(output_dir)

        results_predicted, class_names = self.predict_detect_model(image)

        for class_index, (x_min, y_min, x_max, y_max) in enumerate(results_predicted):
            cropped_img = image[y_min:y_max, x_min:x_max]  # Вырезаем область из изображения
            class_name = class_names[class_index]  # Получаем имя класса

            # Сохраняем вырезанное изображение с уникальным именем
            output_path = os.path.join(output_dir, f"{class_name}_{class_index}.jpg")
            cv2.imwrite(output_path, cropped_img)

            print(f"Saved cropped image {output_path}")

    def create_dataset(self, image, clean_output_dir=False):
        output_dir = 'dataset'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if clean_output_dir:
            delete_files_in_folder(output_dir)

        results_predicted, class_names = self.predict_detect_model(image)
        unique_class_names = np.unique(class_names)
        data_in_row = []
        index_in_stack = set()
        row = -1
        for class_index_1, (x_min_1, y_min_1, x_max_1, y_max_1) in enumerate(results_predicted):
            # Нашли роспись, значит где-то рядом есть имя.
            if class_index_1 not in index_in_stack and class_names[class_index_1] == PredictClass.Signature.value:
                row += 1
                data_in_row.append({unique_class_names[0]: [], unique_class_names[1]: []})
                data_in_row[row][class_names[class_index_1]].append((x_min_1, y_min_1, x_max_1, y_max_1))
                index_in_stack.add(class_index_1)
            else:
                continue

            for class_index_2, (x_min_2, y_min_2, x_max_2, y_max_2) in enumerate(results_predicted):
                if (abs(y_min_2 - y_min_1) < 15  # Находятся на одной прямой
                        and class_index_2 not in index_in_stack  # Не просмотрена
                        and abs(x_max_2 - x_min_1) < 150  # Находятся близко друг к другу
                        and class_names[class_index_2] == PredictClass.FullName.value):  # Это имя
                    data_in_row[row][class_names[class_index_2]].append((x_min_2, y_min_2, x_max_2, y_max_2))
                    index_in_stack.add(class_index_2)

        self.__visualise_result_predicted(image, data_in_row)
        return

    def __visualise_result_predicted(self, image, data):
        # Копия изображения для визуализации
        image_copy = image.copy()

        # Генерируем уникальные цвета для каждого элемента в data
        colors = self.__generate_colors(len(data))

        for idx, item in enumerate(data):
            color = colors[idx]
            image_copy = image.copy()

            for key in item:
                for (x_min, y_min, x_max, y_max) in item[key]:
                    # Рисуем прямоугольник на изображении
                    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
                    # Добавляем текст метки
                    cv2.putText(image_copy, key, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Показываем изображение
            cv2.imshow('Result', image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def __generate_colors(num_colors):
        # Генерируем случайные цвета
        colors = []
        for _ in range(num_colors):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)
        return colors
