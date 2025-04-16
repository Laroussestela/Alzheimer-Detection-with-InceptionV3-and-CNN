import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split


main_path = "E:/00. Kaggle/11. Best Alzheimer's MRI Dataset 99% Accuracy/Combined Dataset/"
train_dir = Path(main_path + 'train')
test_dir = Path(main_path + 'test')

filepaths_train = list(train_dir.glob(r'**/*.jpg'))
labels = [fp.parent.name for fp in filepaths_train]

train_df = pd.DataFrame({'Filepath': filepaths_train, 'Label': labels}).astype(str)
train_df = train_df.sample(frac=1).reset_index(drop = True)

filepaths_test = list(test_dir.glob(r'**/*.jpg'))
labels = [fp.parent.name for fp in filepaths_test]

test_df = pd.DataFrame({'Filepath': filepaths_test, 'Label': labels}).astype(str)
test_df = test_df.sample(frac=1).reset_index(drop = True)

train_df.head(3)
test_df.head(3)


train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["Label"], random_state=42)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


print(len(train_df)) # 8192
print(len(val_df)) # 2048
print(len(test_df)) # 1279


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)
val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(128,128),
    color_mode='rgb',
    class_mode='categorical',
    seed = 42,
    shuffle=False
)

val_images = val_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(128,128),
    color_mode='rgb',
    class_mode='categorical',
    seed = 42,
    shuffle=False
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='categorical',
    seed = 42,
    shuffle=False
)

CLASSES = list(test_images.class_indices.keys())
