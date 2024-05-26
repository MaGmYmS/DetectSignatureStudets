import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def delete_files_in_folder(folder_path):
    """
    TODO: Удаляет все файлы из папки
    :param folder_path:
    :return:
    """

    # Создание папки для сохранения кадров, если она не существует
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file: {file_path} -- {e}")
    print(f"Удалил все файлы в директории {folder_path}")


def auto_annotate_dataset(img_list, model, directory, train_images_path):
    for img_name in tqdm(img_list, desc="Detected frame", unit="frame"):
        img_filepath = directory + "\\" + img_name
        img = cv2.imread(img_filepath)
        img_copy = img

        # получаем ширину и высоту картинки
        h, w, _ = img.shape

        # получаем предсказания по картинке
        results = model.predict(source=img, conf=0.50, verbose=False)

        # расшифровываем объект results
        bboxes_ = results[0].boxes.xyxy.tolist()
        bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
        confs_ = results[0].boxes.conf.tolist()
        confs = list(map(lambda x: int(x * 100), confs_))
        classes_ = results[0].boxes.cls.tolist()
        classes = list(map(lambda x: int(x), classes_))
        cls_dict = results[0].names
        # TODO Косяк с классами
        class_names = list(map(lambda x: cls_dict[x], classes))

        # приводим дешифрированные данные в удобный вид
        annot_lines = []
        for index, val in enumerate(class_names):
            x_min, y_min, x_max, y_max = int(bboxes[index][0]), int(bboxes[index][1]), int(bboxes[index][2]), int(
                bboxes[index][3])
            width = x_max - x_min
            height = y_max - y_min
            center_x = x_min + (width / 2)
            center_y = y_min + (height / 2)
            annotation = f"{classes[index]} {center_x / w} {center_y / h} {width / w} {height / h}"
            annot_lines.append(annotation)

        # копируем картинку в папку базы изображений для импорта
        cv2.imwrite(str(os.path.join(train_images_path, img_name)), img_copy)

        # определить расширение изображения (png или jpg)
        img_extension = os.path.splitext(img_name)[-1].lower()

        # записываем файл аннотации в папку базы изображений для импорта

        txt_name = img_name.replace(img_extension, ".txt")
        with open(os.path.join(r"Data\self_development_dataset/train/labels", txt_name), 'w') as f:
            for line in annot_lines:
                f.write(line)
                f.write('\n')


def main(number_train):
    my_best_model = f"runs/detect/train{number_train}/weights/best.pt"  # Загружаем модель
    model = YOLO(my_best_model)
    # Отсюда берем кадры
    directory = r"Data\self_development_images"
    train_images_path = r"Data\self_development_dataset/train/images"  # Сюда складываем
    img_list = os.listdir(directory)
    print(f"В папке имеется {len(img_list)} изображений")

    delete_files_in_folder(r"Data\self_development_dataset/train/images")
    delete_files_in_folder(r"Data\self_development_dataset/train/labels")
    auto_annotate_dataset(img_list, model, directory, train_images_path)


if __name__ == "__main__":
    main(number_train=30)
