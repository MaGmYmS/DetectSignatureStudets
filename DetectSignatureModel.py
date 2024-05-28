import enum
import os
import random
from difflib import get_close_matches
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

from ultralytics import YOLO
from SelfDevelopment import delete_files_in_folder


def generate_colors(num_colors):
    # Генерируем случайные цвета
    colors = []
    for _ in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors


class PredictClass(enum.Enum):
    FullName = "Full-name"
    Signature = "Signature"


class DetectSignatureModel:
    def __init__(self, number_train):
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
        data_name_signature = []
        index_in_stack = set()
        row = -1
        for class_index_1, (x_min_1, y_min_1, x_max_1, y_max_1) in enumerate(results_predicted):
            # Нашли роспись, значит где-то рядом есть имя.
            if class_index_1 not in index_in_stack and class_names[class_index_1] == PredictClass.Signature.value:
                row += 1
                data_name_signature.append({unique_class_names[0]: [], unique_class_names[1]: []})
                data_name_signature[row][class_names[class_index_1]].append((x_min_1, y_min_1, x_max_1, y_max_1))
                index_in_stack.add(class_index_1)
            else:
                continue

            for class_index_2, (x_min_2, y_min_2, x_max_2, y_max_2) in enumerate(results_predicted):
                if (abs(y_min_2 - y_min_1) < 30  # Находятся на одной прямой
                        and class_index_2 not in index_in_stack  # Не просмотрена
                        and abs(x_max_2 - x_min_1) < 250  # Находятся близко друг к другу
                        and class_names[class_index_2] == PredictClass.FullName.value):  # Это имя
                    data_name_signature[row][class_names[class_index_2]].append((x_min_2, y_min_2, x_max_2, y_max_2))
                    index_in_stack.add(class_index_2)

        if visualise:
            self.__visualise_all_result_predicted(image, data_name_signature)
        # self.__visualise_result_predicted(image, data_in_row)
        print(f"Найдено {len(data_name_signature)} сигнатур")
        return data_name_signature

    def create_dataset_with_signature(self, image_dir=r"Data\data_2", base_dir=r"Data\create_dataset_with_signature",
                                      visualise=False):
        delete_files_in_folder(base_dir)
        # Создаем базовую директорию, если она не существует
        os.makedirs(base_dir, exist_ok=True)

        # Проходим по всем изображениям в папке
        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                print(f"\n\nОбрабатываю изображение {filename}")
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
                closest_match = get_close_matches(text, self.full_name_all_people_ru, n=1, cutoff=0.6)
                if closest_match:
                    closest_name = closest_match[0]
                else:
                    print(f"ФИО \"{text}\" не найдено в списке")
                    closest_name = "Не удалось распознать"

                # print(f"Распознанное имя: {text}, Найденное соответствие: {closest_name}")

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
                cropped_img_gray_signature = cv2.cvtColor(cropped_img_signature, cv2.COLOR_BGR2GRAY)
                save_path = os.path.join(person_dir, f"Signature {next_index}.jpg")
                cv2.imwrite(save_path, cropped_img_gray_signature)
                # print(f"Изображение сохранено по пути: {save_path}\n\n\n")
            else:
                print(f"Потеряли ФИО с подписью")

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
    # endregion


