import contextlib
from difflib import get_close_matches
import io
import cv2
from matplotlib import pyplot as plt
from pytesseract import pytesseract

from DatasetFormer import DatasetFormer
from ultralytics import YOLO
from misc import PredictClass


class ClassifierSignatureModel:
    def __init__(self, number_train=""):
        my_best_model = f"runs/classify/train{number_train}/weights/best.pt"  # Загружаем модель
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

    def __get_result_predict(self, image):
        # Получение предсказания для обрезанного изображения
        with contextlib.redirect_stdout(io.StringIO()):  # Отключение вывода
            results = self.model.predict(image)
        # Извлечение класса с наибольшей вероятностью и уверенности
        probs = results[0].probs
        class_id = probs.top1
        confidence = probs.top1conf
        return results[0].names[class_id], confidence

    def get_result_predict(self, image, data, visualize=False):
        image = DatasetFormer.resize_image(image)
        results_predict_classifier = []
        for idx, item in enumerate(data):
            for (x_min, y_min, x_max, y_max) in item[PredictClass.Signature.value]:
                # Извлечение прямоугольника из изображения
                crop_image = image[y_min:y_max, x_min:x_max]
                # Получение предсказания для обрезанного изображения
                class_name, confidence = self.__get_result_predict(crop_image)

                # Определение реального класса сигнатуры
                if len(item[PredictClass.FullName.value]) > 0:
                    (x_min_name, y_min_name, x_max_name, y_max_name) = item[PredictClass.FullName.value][0]
                    # Вырезаем область с именем из изображения
                    cropped_img_full_name = image[y_min_name:y_max_name, x_min_name:x_max_name]
                    # Распознаем текст на вырезанном изображении
                    text = pytesseract.image_to_string(cropped_img_full_name, lang='rus').strip()
                else:
                    text = "У сигнатуры нет пары-имени"
                # Находим наиболее похожее имя из списка
                closest_match = get_close_matches(text, self.full_name_all_people_ru, n=1, cutoff=0.6)
                if closest_match:
                    closest_name = closest_match[0]
                    class_name_index = self.full_name_all_people_ru.index(closest_name)
                    class_name_en = self.full_name_all_people_en[class_name_index]
                else:
                    print(f"ФИО \"{text}\" не найдено в списке")
                    class_name_en = "Не удалось распознать"

                results_predict_classifier.append(
                    {
                        PredictClass.FullName.value: class_name_en,
                        PredictClass.Signature.value: (x_min, y_min, x_max, y_max),
                        "Predicted": (class_name, confidence)
                    }
                )

        if visualize:
            self.__visualise_all_result_predicted(image, results_predict_classifier)

        return results_predict_classifier

    @staticmethod
    def __interpolate_color(start_color, end_color, factor):
        return tuple(int(start_color[i] + (end_color[i] - start_color[i]) * factor) for i in range(3))

    def __visualise_all_result_predicted(self, image, data):
        # Копия изображения для визуализации
        image_copy = image.copy()

        for idx, item in enumerate(data):
            full_name = item[PredictClass.FullName.value]
            (x_min, y_min, x_max, y_max) = item[PredictClass.Signature.value]
            predicted_class, confidence = item["Predicted"]

            # Если имена совпадают
            if full_name == predicted_class:
                # Интерполируем цвет текста точности от зеленого к красному
                start_color = (0, 0, 255)  # Красный
                end_color = (255, 0, 0)  # Зеленый
                interpolated_color = self.__interpolate_color(start_color, end_color, confidence)

                # Рисуем зеленый прямоугольник и добавляем точность
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image_copy, f"{predicted_class}: {confidence:.2f}",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, interpolated_color, 2)
            else:
                # Рисуем красный прямоугольник и выводим сообщение в консоль
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(image_copy, f"{predicted_class}: {confidence:.2f}",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"Mismatch: Real - {full_name}, Predicted - {predicted_class}, Confidence - {confidence:.2f}")

        # Конвертируем изображение из BGR в RGB для правильного отображения в Matplotlib
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Показываем изображение с помощью Matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(image_copy)
        plt.axis('off')
        plt.show()
