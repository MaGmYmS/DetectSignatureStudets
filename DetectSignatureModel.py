import cv2
import numpy as np
from matplotlib import pyplot as plt

from misc import generate_colors, PredictClass
from ultralytics import YOLO
from DatasetFormer import DatasetFormer


class DetectSignatureModel:
    def __init__(self, number_train=""):
        my_best_model = f"runs/detect/train{number_train}/weights/best.pt"  # Загружаем модель
        model = YOLO(my_best_model)
        self.model = model
        self.full_name_all_people_ru = [
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
            "Шуруев Андрей Вячеславович"]
        self.full_name_all_people_en = [
            "Abdireukov-Samat-Talgatuly",
            "Abushev-Hassan-Ibrahimovich",
            "Anufriev-Nikita-Alexandrovich",
            "Asadulin-Ruslan-Ramilevich",
            "Bessonova-Sofya-Denisovna",
            "Bovykina-Ekaterina-Evgenievna",
            "Vasilets-Anastasia-Artemovna",
            "Vlasov-Sergey-Evgenievich",
            "Galkin-Danil-Vyacheslavovich",
            "Gerasimov-Alexander-Vladimirovich",
            "Giyasov-Nikita-Rushenovich",
            "Golovanov-Sergey-Evgenievich",
            "Gordienko-Andrey-Vyacheslavovich",
            "Gurtovenko-Tatyana-Nikolaevna",
            "Demukhametov-Pavel-Nasipovich",
            "Ershov-Alexander-Andreevich",
            "Zhiryakov-Valentin-Stanislavovich",
            "Zhmenko-Artyom-Yurievich",
            "Zagainova-Evgeniya-Olegovna",
            "Zarembo-Yakov-Andreevich",
            "Zemskov-Nikita-Alexandrovich",
            "Ivanova-Darya-Vladimirovna",
            "Kalimova-Altynai-Yesenbaevna",
            "Kirichenko-Denis-Dmitrievich",
            "Kiseleva-Anastasia-Andreevna",
            "Kobylkina-Anna-Andreevna",
            "Kolesnikov-Dmitry-Andreevich",
            "Kondratiev-Ilya-Vladimirovich",
            "Konyaev-Ilya-Alekseevich",
            "Kopyrin-Evgeny-Alexandrovich",
            "Kuznetsov-Viktor-Vyacheslavovich",
            "Kuzmin-Artyom-Andreevich",
            "Maksimyuk-Alexander-Evgenievich",
            "Marochkina-Victoria-Vitalievna",
            "Netkachev-Pavel-Ivanovich",
            "Nizamov-Artyom-Rustamovich",
            "Potekhin-Ilya-Romanovich",
            "Proschenko-Alexey-Alexandrovich",
            "Sagadeev-Artur-Rinatovich",
            "Serov-Nikita-Sergeyevich",
            "Silin-Nikita-Valeryevich",
            "Strunin-Vitaly-Dmitrievich",
            "Stryukov-Vladislav-Nikolaevich",
            "Titov-Pavel-Sergeevich",
            "Turov-Damir-Alekseevich",
            "Usacheva-Elizaveta-Yurievna",
            "Ustinov-Ilya-Alexandrovich",
            "Friedrich-Alexander-Sergeevich",
            "Chupeev-Andrey-Dmitrievich",
            "Shvetsov-Nikita-Alexandrovich",
            "Sheremet-Arseniy-Andreevich",
            "Shikhova-Anna-Mikhailovna",
            "Shuruev-Andrey-Vyacheslavovich"
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

    def get_result_predict(self, image, visualise=False, visualise_intermediate=False):
        image = DatasetFormer.resize_image(image)
        results_predicted, class_names = self.__predict_detect_model(image)
        unique_class_names = np.unique(class_names)
        data_name_signature = []
        index_in_stack = set()
        row = -1
        image_height, image_width, _ = image.shape

        for class_index_1, (x_min_1, y_min_1, x_max_1, y_max_1) in enumerate(results_predicted):
            # Нашли роспись, значит где-то рядом есть имя.
            if class_index_1 not in index_in_stack and class_names[class_index_1] == PredictClass.Signature.value:
                row += 1
                data_name_signature.append({unique_class_names[0]: [], unique_class_names[1]: []})
                data_name_signature[row][class_names[class_index_1]].append((x_min_1, y_min_1, x_max_1, y_max_1))
                index_in_stack.add(class_index_1)
            else:
                continue

            # Определяем область поиска имени
            search_area_x_min = max(x_min_1 - image_width // 4, 0)
            search_area_y_min = y_min_1
            search_area_x_max = x_min_1
            search_area_y_max = y_max_1

            for class_index_2, (x_min_2, y_min_2, x_max_2, y_max_2) in enumerate(results_predicted):
                if (class_names[class_index_2] == PredictClass.FullName.value
                        and class_index_2 not in index_in_stack):
                    center_full_name_x = (x_min_2 + x_max_2) // 2
                    center_full_name_y = (y_min_2 + y_max_2) // 2
                    # Проверяем, находится ли имя в пределах заданной области
                    if (search_area_x_min <= center_full_name_x <= search_area_x_max and
                            search_area_y_min <= center_full_name_y <= search_area_y_max):
                        # Добавляем найденное имя в массив
                        data_name_signature[row][class_names[class_index_2]].append(
                            (x_min_2, y_min_2, x_max_2, y_max_2))
                        index_in_stack.add(class_index_2)
                        if visualise_intermediate:
                            self.__visualise_intermediate_predicted(image, (x_min_2, y_min_2, x_max_2, y_max_2),
                                                                    (x_min_1, y_min_1, x_max_1, y_max_1))
                        break  # Имя найдено, можно выходить из цикла

        if visualise:
            self.__visualise_all_result_predicted(image, data_name_signature)
        print(f"Найдено {len(data_name_signature)} сигнатур")
        return data_name_signature

    # region Visualise
    @staticmethod
    def __visualise_all_result_predicted(image, data):
        # Копия изображения для визуализации
        image_copy = image.copy()

        # Генерируем уникальные цвета для каждого элемента в data
        colors = generate_colors(len(data))

        for idx, item in enumerate(data):
            color = colors[idx]

            for key in item:
                for (x_min, y_min, x_max, y_max) in item[key]:
                    # Рисуем прямоугольник на изображении
                    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
                    # Добавляем текст метки
                    cv2.putText(image_copy, key, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Конвертируем изображение из BGR в RGB для правильного отображения в Matplotlib
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Показываем изображение с помощью Matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(image_copy)
        plt.axis('off')
        plt.show()

    @staticmethod
    def __visualise_result_predicted(image, data):
        # Генерируем уникальные цвета для каждого элемента в data
        colors = generate_colors(len(data))

        for idx, item in enumerate(data):
            color = colors[idx]
            image_copy = image.copy()

            for key in item:
                for (x_min, y_min, x_max, y_max) in item[key]:
                    # Рисуем прямоугольник на изображении
                    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
                    # Добавляем текст метки
                    cv2.putText(image_copy, key, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Конвертируем изображение из BGR в RGB для правильного отображения в Matplotlib
                image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

                # Показываем изображение с помощью Matplotlib
                plt.figure(figsize=(12, 8))
                plt.imshow(image_copy)
                plt.axis('off')
                plt.show()

    @staticmethod
    def __visualise_intermediate_predicted(image, name_box, signature_box):
        image_copy = image.copy()
        # Извлечение координат рамок
        x_min_name, y_min_name, x_max_name, y_max_name = name_box
        x_min_sig, y_min_sig, x_max_sig, y_max_sig = signature_box

        # Рисуем рамку вокруг имени
        cv2.rectangle(image_copy, (x_min_name, y_min_name), (x_max_name, y_max_name), (0, 255, 0), 2)

        # Рисуем рамку вокруг сигнатуры
        cv2.rectangle(image_copy, (x_min_sig, y_min_sig), (x_max_sig, y_max_sig), (0, 0, 255), 2)

        image_height, image_width, _ = image.shape
        # Находим координаты области слева от сигнатуры
        left_area_x_min = max(x_min_sig - image_width // 4, 0)
        left_area_y_min = y_min_sig
        left_area_x_max = x_min_sig
        left_area_y_max = y_max_sig

        # Рисуем рамку вокруг области слева от сигнатуры
        cv2.rectangle(image_copy, (left_area_x_min, left_area_y_min), (left_area_x_max, left_area_y_max), (255, 0, 0),
                      2)

        # Конвертируем изображение из BGR в RGB для правильного отображения в Matplotlib
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Показываем изображение с помощью Matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(image_copy)
        plt.axis('off')
        plt.show()

    # endregion
