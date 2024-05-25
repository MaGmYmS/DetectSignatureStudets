import cv2
from CreateDetectedSignatureModelTrain import CreateCustomYOLOv8Model
from DetectSignatureModel import DetectSignatureModel


def main():
    # model = CreateCustomYOLOv8Model()
    # model.download_dataset()
    # model.train_my_model(number_epoch=100)

    detect_model = DetectSignatureModel(number_train=29)
    detect_model.create_dataset_with_signature()
    # image_path = r"data 2\Test1_iteration_1.jpg"
    # image = cv2.imread(image_path)
    # detect_model.get_result_predict(image)


if __name__ == "__main__":
    main()
