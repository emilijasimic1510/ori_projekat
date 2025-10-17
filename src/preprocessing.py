from __future__ import annotations
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def make_datagens(img_size=(128, 128)):
    """
    Train datagen sa augmentacijom; val/test samo normalizacija.
    """
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.85, 1.15]
    )
    eval_gen = ImageDataGenerator(rescale=1./255)
    return train_gen, eval_gen

def flow_from_dataframe(
    datagen: ImageDataGenerator,
    df: pd.DataFrame,
    img_size=(128,128),
    batch_size=32,
    shuffle=True
):
    """
    Kreira generator direktno iz DataFrame-a (bez fizičkog kopiranja slika u class-foldere).
    DF mora da ima kolone: filepath, label.
    """
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col="label",
        target_size=img_size,
        class_mode="sparse",   # ili "categorical" ako želiš one-hot
        batch_size=batch_size,
        shuffle=shuffle
    )
