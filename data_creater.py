from DatasetFormer import DatasetFormer
from DetectSignatureModel import DetectSignatureModel


def main(resize=False, data_with_signature=False, augmentation=False, number_images_in_one_class=None,
         result_dataset_create=False):
    model = DetectSignatureModel()
    dataset_former = DatasetFormer(detect_model=model)
    if resize:
        print("Изменяю размер входных изображений на 1920х1080")
        input_folder = r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data\data_2"
        output_folder = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data"
                         r"\resized_images")
        dataset_former.resize_images_in_folder(input_folder, output_folder)

    if data_with_signature:
        print("Извлекаю данные подписей студентов из изображения...")
        dataset_former.create_dataset_with_signature()

    if augmentation:
        print("Аугментирую данные, это займет какое-то время...")
        dataset_former.create_augmentation_folder(angle_value=10, shift_value=10, resize_value=64)

    if number_images_in_one_class:
        print(f"Равномерно выбираю {number_images_in_one_class} изображений из каждого класса")
        dataset_former.copy_number_images_to_new_folder(number_images_in_one_class)

    if result_dataset_create:
        print("Формирую выборки train, valid, test...")
        source_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data"
                      r"\distributed_dataset_with_signature_augmented")
        dest_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\yolov5\datasets"
                    r"\final_distributed_dataset_with_signature_augmented")
        dataset_former.split_dataset(source_dir, dest_dir)


if __name__ == "__main__":
    main(result_dataset_create=True)
