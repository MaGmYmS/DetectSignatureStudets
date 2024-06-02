import enum
import random


def generate_colors(num_colors):
    # Генерируем случайные цвета
    colors = []
    for _ in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors


class PredictClass(enum.Enum):
    FullName = "Full-name"
    Signature = "Signature"
