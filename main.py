import cv2
from CreateDetectedSignatureModelTrain import CreateCustomYOLOv8Model
from DetectSignatureModel import DetectSignatureModel


def main():
    # model = CreateCustomYOLOv8Model()
    # model.download_dataset()
    # model.train_my_model(number_epoch=100)

    detect_model = DetectSignatureModel(number_train=29)
    image_path = r"self development images\Test1.jpg"
    image = cv2.imread(image_path)
    detect_model.create_dataset(image)


if __name__ == "__main__":
    main()
