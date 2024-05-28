from ultralytics import YOLO


class ClassifierSignatureModel:
    def __init__(self, number_train):
        my_best_model = f"runs/classify/train{number_train}/weights/best.pt"  # Загружаем модель
        model = YOLO(my_best_model)
        self.model = model

    def __get_result_predict(self, image):
        # Получение предсказания для обрезанного изображения
        results = self.model.predict(image)
        # Извлечение класса с наибольшей вероятностью и уверенности
        probs = results[0].probs
        class_id = probs.top1
        confidence = probs.top1conf
        return results[0].names[class_id], confidence

    def get_result_predict(self, image, data):
        results = []
        for idx, item in enumerate(data):
            for key in item:
                for (x_min, y_min, x_max, y_max) in item[key]:
                    # Извлечение прямоугольника из изображения
                    crop_image = image[y_min:y_max, x_min:x_max]
                    # Получение предсказания для обрезанного изображения
                    class_name, confidence = self.__get_result_predict(crop_image)
                    results.append((class_name, confidence))
        return results
