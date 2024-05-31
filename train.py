from CreateDetectedSignatureModelTrain import CreateCustomYOLOv8Model, TypeModel


def train_model_detect_signature_and_full_name():
    model = CreateCustomYOLOv8Model(dataset_version=3)
    model.download_dataset()
    model.train_my_model(model_size="s", number_epoch=50, image_size=1280, type_model=TypeModel.Detect)


def train_model_classifier_signature():
    model = CreateCustomYOLOv8Model()
    data_to_dataset = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\yolov5\datasets"
                       r"\final_final_dataset_with_signature_augmented_10000")
    model.train_my_model(model_size="n", number_epoch=1, image_size=64, path_to_data=data_to_dataset,
                         type_model=TypeModel.Classification)


def main(type_my_model: TypeModel):
    if type_my_model == TypeModel.Detect:
        train_model_detect_signature_and_full_name()
    else:
        train_model_classifier_signature()


if __name__ == "__main__":
    type_model = TypeModel.Detect
    main(type_model)
