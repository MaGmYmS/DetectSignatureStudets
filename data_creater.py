from DatasetFormer import DatasetFormer


def main(resize=False, augmentation=False, number_images_in_one_class=None, result_dataset_create=False):
    dataset_former = DatasetFormer()
    if resize:
        input_folder = r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data\data_2"
        output_folder = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data"
                         r"\resized_images")
        dataset_former.resize_images_in_folder(input_folder, output_folder)

    if augmentation:
        dataset_former.create_augmentation_folder(angle_value=10, shift_value=10, resize_value=64)

    if number_images_in_one_class:
        dataset_former.copy_number_images_to_new_folder(number_images_in_one_class)

    if result_dataset_create:
        source_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов Data\Data"
                      r"\final_dataset_with_signature_augmented_10000")
        dest_dir = (r"D:\я у мамы программист\3 курс 2 семестр КЗ\Распознавание подписей студентов\yolov5\datasets"
                    r"\final_final_dataset_with_signature_augmented_10000")
        dataset_former.split_dataset(source_dir, dest_dir)


if __name__ == "__main__":
    main()
