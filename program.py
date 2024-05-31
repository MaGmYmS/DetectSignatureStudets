import cv2

from ClassifierSignatureModel import ClassifierSignatureModel
from DetectSignatureModel import DetectSignatureModel


def main():
    detect_model = DetectSignatureModel(number_train=33)
    classifier_model = ClassifierSignatureModel(number_train="")
    image_path = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data\data_2"
                  r"\Test1_iteration_1.jpg")
    image = cv2.imread(image_path)
    detect_model.get_result_predict(image, visualise=True)

    data = detect_model.get_result_predict(image)
    classification_signatures = classifier_model.get_result_predict(image, data, visualize=True)
    print(classification_signatures)


if __name__ == "__main__":
    main()
