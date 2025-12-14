import tensorflow as tf
import kagglehub
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download Dataset 

DATASET_PATH = r"C:\Users\A.M\Desktop\New folder\dataset_root"


TRAIN_DIR = f"{DATASET_PATH}/train"
TEST_DIR  = f"{DATASET_PATH}/test"

print("Dataset path:", DATASET_PATH)


#  Image Generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    classes=["NORMAL", "PNEUMONIA"], 
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    classes=["NORMAL", "PNEUMONIA"],
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    classes=["NORMAL", "PNEUMONIA"],
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


#  Base Model 

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# FREEZE base model
base_model.trainable = False

# Full Model

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])


#  Compile

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#  Train

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


#  Test Accuracy
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")


#  Save Model

model.save("pneumonia_model.h5")
print("Model saved as pneumonia_model.h5")
