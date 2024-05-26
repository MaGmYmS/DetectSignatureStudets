import cv2
from CreateDetectedSignatureModelTrain import CreateCustomYOLOv8Model
from DetectSignatureModel import DetectSignatureModel


def main():
    # model = CreateCustomYOLOv8Model(dataset_version=3)
    # model.download_dataset()
    # model.train_my_model(model_size="s", number_epoch=50, image_size=1280)
    # model.train_my_model(model_size="s", number_epoch=100, image_size=64)

    detect_model = DetectSignatureModel(number_train=33)
    # input_folder = r"Data\data_2"
    # output_folder = r"Data\resized_images"
    # detect_model.resize_images_in_folder(input_folder, output_folder)

    # detect_model.create_dataset_with_signature(visualise=False, image_dir=r"Data\resized_images")
    # image_path = r"Data\resized_images\Test10_iteration_1.jpg"
    # image = cv2.imread(image_path)
    # detect_model.get_result_predict(image, visualise=True)

    # detect_model.create_augmentation_folder(angle_value=10, shift_value=10, resize_value=64)
    detect_model.copy_images_to_new_folder(10000)


if __name__ == "__main__":
    main()
