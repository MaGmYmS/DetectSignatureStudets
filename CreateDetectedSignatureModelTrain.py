import enum
import os
import random
import shutil

import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class TypeModel(enum.Enum):
    Detect = enum.auto()
    Classification = enum.auto()


class CreateCustomYOLOv8Model:
    def __init__(self, dataset_version=None):
        self.dataset_version = dataset_version
        self.rf = Roboflow(api_key="AmQ0vHqaiNHr6SeXUAWb")
        self.workspace_name = "detected"
        self.dataset_name = "signatures-of-moiais-students-3"

    # region чтение датасета
    @staticmethod
    def _update_data_train_yaml(data_yaml_path):
        """
        Обновляет файл данных YAML для указания путей к изображениям обучения и валидации.

        :param data_yaml_path: Путь к файлу данных YAML.
        :type data_yaml_path: str
        :return: Нет возвращаемого значения.
        """
        train_path = "../train/images"
        val_path = "../valid/images"

        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        data['train'] = train_path
        data['val'] = val_path

        with open(data_yaml_path, 'w') as file:
            yaml.dump(data, file)

    def __download_datasets_from_roboflow(self):
        """
        Загружает датасет с Roboflow

        :return: путь к датасету
        :rtype: str
        """
        project = self.rf.workspace(self.workspace_name).project(self.dataset_name)
        version = project.version(self.dataset_version)
        dataset = version.download("yolov8")
        return dataset.location

    @staticmethod
    def __delete_exists_folder(folder_path):
        """
        Удаляет папку, если она существует

        :param folder_path: Путь к папке
        :type folder_path: str
        """
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f'Папка {folder_path} успешно удалена.')

    def download_dataset(self):
        """
        Загружает датасет с Roboflow и организует его структуру.

        Качает датасет с использованием API-ключа Roboflow. Перемещает файлы из исходной папки в целевую,
        а затем обновляет файл данных YAML с новыми путями.

        :return: Нет возвращаемого значения.
        """
        if self.dataset_version is None:
            print("При загрузке данных произошла ошибка. Версия датасета не определена")
            return
        dataset_path = self.dataset_name + "-" + str(self.dataset_version)
        self.__delete_exists_folder(dataset_path)
        target_folder = os.path.join("yolov5", "datasets", dataset_path)
        self.__delete_exists_folder(target_folder)

        source_folder = self.__download_datasets_from_roboflow()
        data_yaml_path = f"{dataset_path}/data.yaml"
        self._update_data_train_yaml(data_yaml_path)

        # Если папка не существует, создайте ее
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Переместите файлы в целевую папку
        for file_name in os.listdir(source_folder):
            source_file_path = os.path.join(source_folder, file_name)
            target_file_path = os.path.join(target_folder, file_name)
            print(f"Перенос файла {file_name}")
            shutil.move(source_file_path, target_file_path)

        # Удаляем папку, если она пуста
        try:
            os.rmdir(dataset_path)
            print(f'Папка {dataset_path} успешно удалена.')
        except OSError as e:
            print(f'Не удалось удалить папку {dataset_path}: {e}')

        self.__download_datasets_from_roboflow()
        self._update_data_train_yaml(data_yaml_path)

    @staticmethod
    def __update_work_directory_in_data_yaml(path_data_yaml_settings=r'C:\Users\user\AppData\Roaming\Ultralytics'
                                                                     r'\settings.yaml', path_data_yaml_dataset=None):
        """
        Меняем рабочую директорию в YAML файле для обучения.
        :param path_data_yaml_settings: Путь к файлу данных YAML.
        :type path_data_yaml_settings: str
        :return: Нет возвращаемого значения.
        """

        # Получаем текущую директорию
        current_directory = os.getcwd()
        print("Current Directory:", current_directory)

        # Формируем путь к директории datasets
        if path_data_yaml_dataset is None:
            directory_path = os.path.join(current_directory, "yolov5", "datasets")
        else:
            directory_path = os.path.join(current_directory, path_data_yaml_dataset)

        # Открываем и читаем YAML файл
        with open(path_data_yaml_settings, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Обновляем значение datasets_dir
        data['datasets_dir'] = directory_path

        # Сохраняем изменения в YAML файл
        with open(path_data_yaml_settings, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, allow_unicode=True)

    # endregion

    def train_my_model(self, model_size="n", number_epoch=50, image_size=640, path_to_data=None,
                       path_data_yaml_dataset=None, type_model=TypeModel.Detect):
        """
        Обучает модель YOLOv8 на предоставленных данных.

        Инициализирует и обучает модель YOLOv8 на указанных данных.
        В данном случае, обучение происходит на протяжении 60 эпох с размером изображения 640x640.

        :return: Нет возвращаемого значения.
        """
        self.__update_work_directory_in_data_yaml(path_data_yaml_dataset=path_data_yaml_dataset)

        if type_model == TypeModel.Detect:
            name_model = f"yolov8{model_size}.pt"
        elif type_model == TypeModel.Classification:
            name_model = f"yolov8{model_size}-cls.pt"
        else:
            name_model = "yolov8n.pt"

        train_model = YOLO(name_model)
        if path_to_data is None:
            if self.dataset_version is not None:
                path_to_data = f"{self.dataset_name}-{self.dataset_version}/data.yaml"
            else:
                print("Обучение прервано, произошла ошибка. Версия датасета не определена")
                return
        print(f"{self.dataset_name}-{self.dataset_version}/data.yaml")
        print(path_to_data)
        train_model.train(data=path_to_data, epochs=number_epoch, imgsz=image_size)
