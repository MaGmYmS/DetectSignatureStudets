import enum
import os
import random
from difflib import get_close_matches
import pytesseract
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
        self.full_name_all_people = [
            "Абдиреуков Самат Талгатулы",
            "Абушев Хасан Ибрагимович",
            "Ануфриев Никита Александрович",
            "Асадулин Руслан Рамилевич",
            "Бессонова Софья Денисовна",
            "Бовыкина Екатерина Евгеньевна",
            "Василец Анастасия Артемовна",
            "Власов Сергей Евгеньевич",
            "Галкин Данил Вячеславович",
            "Герасимов Александр Владимирович",
            "Гиясов Никита Рушенович",
            "Голованов Сергей Евгеньевич",
            "Гордиенко Андрей Вячеславович",
            "Гуртовенко Татьяна Николаевна",
            "Демухаметов Павел Насипович",
            "Ершов Александр Андреевич",
            "Жиряков Валентин Станиславович",
            "Жменько Артём Юрьевич",
            "Загайнова Евгения Олеговна",
            "Зарембо Яков Андреевич",
            "Земсков Никита Александрович",
            "Иванова Дарья Владимировна",
            "Калимова Алтынай Есенбаевна",
            "Кириченко Денис Дмитриевич",
            "Киселева Анастасия Андреевна",
            "Кобылкина Анна Андреевна",
            "Колесников Дмитрий Андреевич",
            "Кондратьев Илья Владимирович",
            "Коняев Илья Алексеевич",
            "Копырин Евгений Александрович",
            "Кузнецов Виктор Вячеславович",
            "Кузьмин Артём Андреевич",
            "Максимюк Александр Евгеньевич",
            "Марочкина Виктория Витальевна",
            "Неткачев Павел Иванович",
            "Низамов Артем Рустамович",
            "Потехин Илья Романович",
            "Прощенко Алексей Александрович",
            "Сагадеев Артур Ринатович",
            "Серов Никита Сергеевич",
            "Силин Никита Валерьевич",
            "Струнин Виталий Дмитриевич",
            "Стрюков Владислав Николаевич",
            "Титов Павел Сергеевич",
            "Туров Дамир Алексеевич",
            "Усачёва Елизавета Юрьевна",
            "Устинов Илья Александрович",
            "Фридрих Александр Сергеевич",
            "Чупеев Андрей Дмитриевич",
            "Швецов Никита Александрович",
            "Шеремет Арсений Андреевич",
            "Шихова Анна Михайловна",
            "Шуруев Андрей Вячеславович"
        ]

    def __predict_detect_model(self, image):
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

    def get_result_predict(self, image, visualise=False):
        results_predicted, class_names = self.__predict_detect_model(image)
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

        if visualise:
            self.__visualise_all_result_predicted(image, data_in_row)
        # self.__visualise_result_predicted(image, data_in_row)
        print(f"Найдено {len(data_in_row)} сигнатур")
        return data_in_row

    def create_dataset_with_signature(self, image_dir=r"Data\data_2", base_dir=r"Data\create_dataset_with_signature",
                                      visualise=False):
        delete_files_in_folder(base_dir)
        # Создаем базовую директорию, если она не существует
        os.makedirs(base_dir, exist_ok=True)

        # Проходим по всем изображениям в папке
        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                print(f"Обрабатываю изображение {filename}\n")
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                # Проверяем, удалось ли загрузить изображение
                if image is not None:
                    result_predict = self.get_result_predict(image, visualise=visualise)
                    self.__create_dataset_with_signature(image, data=result_predict, base_dir=base_dir)
                else:
                    print(f"Не удалось загрузить изображение: {image_path}")

    def __create_dataset_with_signature(self, image, data, base_dir="create_dataset_with_signature"):
        for idx, item in enumerate(data):
            if len(item[PredictClass.FullName.value]) > 0:
                (x_min, y_min, x_max, y_max) = item[PredictClass.FullName.value][0]
                cropped_img_full_name = image[y_min:y_max, x_min:x_max]  # Вырезаем область из изображения

                # Распознаем текст на вырезанном изображении
                text = pytesseract.image_to_string(cropped_img_full_name, lang='rus').strip()

                # Находим наиболее похожее имя из списка
                closest_match = get_close_matches(text, self.full_name_all_people, n=1, cutoff=0.6)
                if closest_match:
                    closest_name = closest_match[0]
                else:
                    closest_name = "Не удалось распознать"

                print(f"Распознанное имя: {text}, Найденное соответствие: {closest_name}")

                # Создаем папку для данного человека
                person_dir = os.path.join(base_dir, closest_name)
                os.makedirs(person_dir, exist_ok=True)

                # Определяем максимальный текущий индекс в папке
                existing_files = os.listdir(person_dir)
                indices = [int(f.split()[1].split('.')[0]) for f in existing_files if f.startswith('Signature ')]
                next_index = max(indices, default=0) + 1

                # Сохранение изображения подписи
                (x_min, y_min, x_max, y_max) = item[PredictClass.Signature.value][0]
                cropped_img_signature = image[y_min:y_max, x_min:x_max]  # Вырезаем область из изображения
                save_path = os.path.join(person_dir, f"Signature {next_index}.jpg")
                cv2.imwrite(save_path, cropped_img_signature)
                print(f"Изображение сохранено по пути: {save_path}\n\n\n")
            else:
                print("ФИО не найдено")

    # region Visualise
    def __visualise_all_result_predicted(self, image, data):
        # Копия изображения для визуализации
        image_copy = image.copy()

        # Генерируем уникальные цвета для каждого элемента в data
        colors = self.__generate_colors(len(data))

        for idx, item in enumerate(data):
            color = colors[idx]

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

    def __visualise_result_predicted(self, image, data):
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
    # endregion

    @staticmethod
    def __resize_image(image, target_width, target_height):
        """
        Изменяет размер изображения до указанных ширины и высоты.

        :param image: исходное изображение (numpy массив)
        :param target_width: целевая ширина
        :param target_height: целевая высота
        :return: изображение с измененным размером
        """
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def resize_images_in_folder(self, input_folder, output_folder, target_width=1920, target_height=1080):
        """
        Изменяет размер всех изображений в папке до указанных ширины и высоты и сохраняет их в другую папку.

        :param input_folder: путь к папке с исходными изображениями
        :param output_folder: путь к папке для сохранения измененных изображений
        :param target_width: целевая ширина (по умолчанию 1920)
        :param target_height: целевая высота (по умолчанию 1080)
        """
        # Создаем выходную папку, если она не существует
        os.makedirs(output_folder, exist_ok=True)

        # Проходим по всем файлам в входной папке
        for filename in os.listdir(input_folder):
            # Полный путь к файлу
            input_path = os.path.join(input_folder, filename)

            # Проверяем, является ли файл изображением
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    # Считываем изображение с помощью OpenCV
                    image = cv2.imread(input_path)

                    # Проверяем, удалось ли загрузить изображение
                    if image is not None:
                        # Изменяем размер изображения
                        resized_image = self.__resize_image(image, target_width, target_height)

                        # Полный путь к выходному файлу
                        output_path = os.path.join(output_folder, filename)

                        # Сохраняем измененное изображение
                        cv2.imwrite(output_path, resized_image)
                        print(f"Изображение сохранено по пути: {output_path}")
                    else:
                        print(f"Не удалось загрузить изображение: {input_path}")
                except Exception as e:
                    print(f"Не удалось обработать файл {input_path}: {e}")