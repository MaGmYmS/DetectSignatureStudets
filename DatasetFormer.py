import random
import os
import shutil
from difflib import get_close_matches

import cv2
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array
from pytesseract import pytesseract
from tqdm import tqdm

from DetectSignatureModel import PredictClass
from SelfDevelopment import delete_files_in_folder


class DatasetFormer:
    def __init__(self, detect_model=None):
        self.detect_model = detect_model
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

        # region Augmentation

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
                    if self.detect_model is not None:
                        result_predict = self.detect_model.get_result_predict(image, visualise=visualise)
                        self.__create_dataset_with_signature(image, data=result_predict, base_dir=base_dir)
                    else:
                        print("Модель не была загружена")
                        return
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

    @staticmethod
    def __rotate_image(image, angle):
        # Функция для поворота изображения
        h, w = image.shape[:2]
        # Создаем пустое белое поле размером 2h и 2w
        bg = np.ones((2 * h, 2 * w, 3), dtype=np.uint8) * 255
        # Помещаем исходное изображение в центр
        bg[h // 2:h // 2 + h, w // 2:w // 2 + w] = image
        center = (w, h)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Применяем поворот
        rotated = cv2.warpAffine(bg, matrix, (2 * w, 2 * h))
        # Обрезаем до размера h и w
        rotated_cropped = rotated[h // 2:h // 2 + h, w // 2:w // 2 + w]
        return rotated_cropped

    @staticmethod
    def __shift_image(image, shift):
        # Функция для сдвига изображения
        h, w = image.shape[:2]
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

        # Создаем белое поле
        white_background = np.full((h, w, 3), 255, dtype=np.uint8)

        # Сдвигаем изображение
        shifted = cv2.warpAffine(image, matrix, (w, h))

        # Найти границы изображения
        _, binary = cv2.threshold(cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            shifted_cropped = shifted[y:y + h, x:x + w]
            white_background[y:y + h, x:x + w] = shifted_cropped

        return white_background

    def create_augmentation_folder(self, angle_value=10, shift_value=10, resize_value=64,
                                   source_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей "
                                              r"студентов Data\Data\create_dataset_with_signature",
                                   augmented_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей "
                                                 r"студентов Data\Data\create_dataset_with_signature_augmented"):
        delete_files_in_folder(augmented_dir)
        # Аугментации
        angles = range(-angle_value, angle_value, 1)
        shifts = [(shift_x, shift_y) for shift_x in range(-shift_value, shift_value + 1, 2) for
                  shift_y in range(-shift_value, shift_value + 1, 2)]
        # Создаем дерево каталогов и выполняем аугментации
        for person in tqdm(self.full_name_all_people_ru):
            person_source_dir = os.path.join(source_dir, person)
            person_augmented_dir = os.path.join(augmented_dir, person)
            # Проверяем, существует ли папка для этого человека
            if not os.path.exists(person_source_dir):
                # print(f"Папка для {person} не существует в директории {source_dir}. Пропускаем.")
                continue
            os.makedirs(person_augmented_dir, exist_ok=True)

            for filename in os.listdir(person_source_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):

                    image_path = os.path.join(person_source_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:

                        # Аугментации: повороты и сдвиги
                        for angle in angles:
                            rotated_image = self.__rotate_image(image, angle)
                            for shift in shifts:
                                rotated_image_copy = rotated_image.copy()
                                rotated_shifted_image = self.__shift_image(rotated_image_copy, shift)
                                resize_rotated_shifted_image = self.__resize_image(rotated_shifted_image,
                                                                                   target_width=resize_value,
                                                                                   target_height=resize_value)
                                shift_label = f"{shift[0]}_{shift[1]}"
                                new_filename = f"{os.path.splitext(filename)[0]}_rot{angle}__shift{shift_label}.jpg"
                                cv2.imwrite(os.path.join(person_augmented_dir, new_filename),
                                            resize_rotated_shifted_image)

        print("Аугментация завершена.")
        self.count_files_in_folders()

    def augment_images_with_ImageDataGenerator(self, num_augmentations=1000,
                                               source_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ"
                                                          r"\Распознавание подписей студентов Data\Data"
                                                          r"\create_dataset_with_signature",
                                               augmented_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ"
                                                             r"\Распознавание подписей студентов Data\Data"
                                                             r"\create_dataset_with_signature_augmented"):
        delete_files_in_folder(augmented_dir)
        # Настройка генератора изображений
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        for person in tqdm(self.full_name_all_people_ru):
            person_source_dir = os.path.join(source_dir, person)
            person_augmented_dir = os.path.join(augmented_dir, person)

            # Проверяем, существует ли папка для этого человека
            if not os.path.exists(person_source_dir):
                # print(f"Папка для {person} не существует в директории {source_dir}. Пропускаем.")
                continue

            os.makedirs(person_augmented_dir, exist_ok=True)

            for filename in os.listdir(person_source_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_source_dir, filename)
                    image = load_img(image_path)
                    x = img_to_array(image)
                    x = x.reshape((1,) + x.shape)

                    i = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=person_augmented_dir, save_prefix='aug',
                                              save_format='jpg'):
                        i += 1
                        if i >= num_augmentations:
                            break
        print("Аугментация завершена.")
        self.count_files_in_folders()

    def resize_images_in_folder(self, input_folder,
                                output_folder=r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей "
                                              r"студентов Data\Data\resized_images", target_width=1920,
                                target_height=1080):
        """
        Изменяет размер всех изображений в папке до указанных ширины и высоты и сохраняет их в другую папку.

        :param input_folder: путь к папке с исходными изображениями
        :param output_folder: путь к папке для сохранения измененных изображений
        :param target_width: целевая ширина (по умолчанию 1920)
        :param target_height: целевая высота (по умолчанию 1080)
        """
        # Создаем выходную папку, если она не существует
        os.makedirs(output_folder, exist_ok=True)
        delete_files_in_folder(output_folder)

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

    def count_files_in_folders(self,
                               augmented_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей "
                                             r"студентов Data\Data\create_dataset_with_signature_augmented"):
        for person in self.full_name_all_people_ru:
            person_dir = os.path.join(augmented_dir, person)
            if os.path.exists(person_dir) and os.path.isdir(person_dir):
                num_files = len([name for name in os.listdir(person_dir) if
                                 os.path.isfile(os.path.join(person_dir, name))])
                print(f"Папка '{person}' содержит {num_files} файлов.")
            else:
                print(f"Папка '{person}' не существует в директории {augmented_dir}.")
        print("\n\n")

    def copy_number_images_to_new_folder(self, number_images,
                                         source_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ"
                                                    r"\Распознавание подписей студентов Data\Data"
                                                    r"\create_dataset_with_signature_augmented",
                                         final_dir=r"D:\я у мамы программист\3 курс 2 семестр КЗ"
                                                   r"\Распознавание подписей студентов Data\Data"
                                                   r"\final_dataset_with_signature_augmented"):
        os.makedirs(final_dir, exist_ok=True)
        delete_files_in_folder(final_dir)
        count_person = len(self.full_name_all_people_ru)
        for i, person in enumerate(self.full_name_all_people_ru):
            person_source_dir = os.path.join(source_dir, person)
            person_final_dir = os.path.join(final_dir, self.full_name_all_people_en[i])

            if os.path.exists(person_source_dir) and os.path.isdir(person_source_dir):
                os.makedirs(person_final_dir, exist_ok=True)
                all_files = [name for name in os.listdir(person_source_dir) if
                             os.path.isfile(os.path.join(person_source_dir, name))]
                num_files = len(all_files)

                if num_files > 0:
                    indices = np.linspace(0, num_files - 1, min(number_images, num_files), dtype=int)
                    selected_files = [all_files[i] for i in indices]

                    for file in selected_files:
                        source_file = os.path.join(person_source_dir, file)
                        destination_file = os.path.join(person_final_dir, file)
                        shutil.copy2(source_file, destination_file)

                    print(f"({i}/{count_person}) Скопировано {len(selected_files)} файлов для {person}.")
                else:
                    print(f"({i}/{count_person}) Нет файлов для копирования в папке {person}.")
            else:
                print(f"({i}/{count_person}) Папка '{person}' не существует в директории {source_dir}.")
        print("\n\n")

    @staticmethod
    def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert train_ratio + val_ratio + test_ratio == 1, "Соотношения должны суммироваться до 1"
        delete_files_in_folder(dest_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        train_dir = os.path.join(dest_dir, 'train')
        val_dir = os.path.join(dest_dir, 'valid')
        test_dir = os.path.join(dest_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        class_names = []
        for class_name in tqdm(os.listdir(source_dir)):
            class_names.append(class_name)
            class_dir = os.path.join(source_dir, class_name)

            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                random.shuffle(images)

                train_split = int(train_ratio * len(images))
                val_split = int(val_ratio * len(images))

                train_images = images[:train_split]
                val_images = images[train_split:train_split + val_split]
                test_images = images[train_split + val_split:]

                for image_set, subset_dir in zip([train_images, val_images, test_images],
                                                 [train_dir, val_dir, test_dir]):
                    subset_class_dir = os.path.join(subset_dir, class_name)
                    os.makedirs(subset_class_dir, exist_ok=True)

                    for image in image_set:
                        src_image_path = os.path.join(class_dir, image)
                        dest_image_path = os.path.join(subset_class_dir, image)
                        shutil.copy2(src_image_path, dest_image_path)
        print("Разделение данных завершено. Датасет успешно создан")

