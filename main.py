import cv2

from ClassifierSignatureModel import ClassifierSignatureModel
from CreateDetectedSignatureModelTrain import CreateCustomYOLOv8Model
from DetectSignatureModel import DetectSignatureModel


def main():
    model = CreateCustomYOLOv8Model(dataset_version=3)
    # model.download_dataset()
    # model.train_my_model(model_size="s", number_epoch=50, image_size=1280)
    # path_data_yaml_dataset = r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\Data"
    # data_to_yaml = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\yolov5\datasets"
    #                 r"\final_final_dataset_with_signature_augmented_10000")
    # model.train_my_model(model_size="n", number_epoch=1, image_size=64, path_to_data=data_to_yaml, type_predict="-cls")

    detect_model = DetectSignatureModel(number_train=33)
    # input_folder = r"Data\data_2"
    # output_folder = r"Data\resized_images"
    # detect_model.resize_images_in_folder(input_folder, output_folder)

    # detect_model.create_dataset_with_signature(visualise=False, image_dir=r"Data\resized_images")
    # image_path = r"Data\resized_images\Test10_iteration_1.jpg"
    # image = cv2.imread(image_path)
    # detect_model.get_result_predict(image, visualise=True)

    # detect_model.create_augmentation_folder(angle_value=10, shift_value=10, resize_value=64)
    # detect_model.copy_images_to_new_folder(10)
    # source_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data"
    #               r"\final_dataset_with_signature_augmented_10000")
    # dest_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\yolov5\datasets"
    #             r"\final_final_dataset_with_signature_augmented_10000")
    # detect_model.split_dataset(source_dir, dest_dir)

    classifier_model = ClassifierSignatureModel(number_train="")
    image_path = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data"
                  r"\Data\data_2\Test1_iteration_1.jpg")
    image = cv2.imread(image_path)
    data = detect_model.get_result_predict(image)
    classification_signatures = classifier_model.get_result_predict(image, data)
    print(classification_signatures)


if __name__ == "__main__":
    main()
