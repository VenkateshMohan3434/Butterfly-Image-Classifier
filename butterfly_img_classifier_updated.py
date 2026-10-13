import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# --- 1. CONFIGURATION ---
IMAGE_DIR = r"C:/Datasets/butterfly_imgs/train"
CSV_PATH = r"C:/Datasets/butterfly_imgs/Training_set.csv"
MODEL_PATH = "best_butterfly_model.keras"

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 15
RANDOM_SEED = 42
NUM_TRAIN_IMAGES_TO_SHOW = 12
NUM_TEST_PREDICTIONS_TO_SHOW = 12


# --- 2. LOAD DATA AND CREATE LABEL MAPPING ---
df = pd.read_csv(CSV_PATH)

required_columns = {"filename", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns {sorted(required_columns)}. "
        f"Found: {list(df.columns)}"
    )

df["filename"] = df["filename"].apply(lambda name: os.path.join(IMAGE_DIR, name))
df["label"] = df["label"].astype(str)

# Keep a stable class-to-index mapping for training and inference.
class_names = sorted(df["label"].unique().tolist())
class_to_index = {name: index for index, name in enumerate(class_names)}
index_to_class = {index: name for name, index in class_to_index.items()}
df["label_code"] = df["label"].map(class_to_index).astype("int32")
NUM_CLASSES = len(class_names)

missing_files = [path for path in df["filename"] if not os.path.isfile(path)]
if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files)} image file(s) listed in the CSV were not found. "
        f"Example: {missing_files[0]}"
    )

print(f"Loaded {len(df)} images across {NUM_CLASSES} butterfly classes.")


# --- 3. SHOW CLASS FREQUENCIES ---
class_counts = df["label"].value_counts().reindex(class_names, fill_value=0)

plt.figure(figsize=(max(12, NUM_CLASSES * 0.45), 6))
class_counts.plot(kind="bar", color="steelblue")
plt.title("Frequency of Each Butterfly Class")
plt.xlabel("Butterfly class")
plt.ylabel("Number of images")
plt.xticks(rotation=70, ha="right")
plt.tight_layout()
plt.show()


# --- 4. SPLIT DATA ---
train_df, test_df = train_test_split(
    df,
    test_size=1000,
    random_state=RANDOM_SEED,
    stratify=df["label_code"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Training images: {len(train_df)}")
print(f"Test/demo images: {len(test_df)}")


# --- 5. SHOW A SAMPLE OF THE TRAINING SET ---
def show_training_samples(dataframe, number_to_show=12):
    sample = dataframe.sample(
        n=min(number_to_show, len(dataframe)),
        random_state=RANDOM_SEED
    )
    columns = 4
    rows = math.ceil(len(sample) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sample.iterrows()):
        image = plt.imread(row["filename"])
        ax.imshow(image)
        ax.set_title(row["label"], fontsize=9)
        ax.axis("off")

    fig.suptitle("Sample Images from the Training Set", fontsize=16)
    plt.tight_layout()
    plt.show()


show_training_samples(train_df, NUM_TRAIN_IMAGES_TO_SHOW)


# --- 6. TF.DATA PIPELINES ---
def load_and_preprocess_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["filename"].values, train_df["label_code"].values)
)
train_dataset = train_dataset.shuffle(
    buffer_size=min(1000, len(train_df)), seed=RANDOM_SEED
)
train_dataset = train_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df["filename"].values, test_df["label_code"].values)
)
test_dataset = test_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- 7. BUILD MODEL (ONLY IF A CHECKPOINT DOES NOT ALREADY EXIST) ---
def build_model():
    augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ], name="data_augmentation")

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if os.path.exists(MODEL_PATH):
    print(f"\nFound saved model: {MODEL_PATH}")
    print("Loading it instead of training again...")
    model = load_model(MODEL_PATH)
else:
    print("\nNo saved model found. Building and training a new model...")
    model = build_model()

    def lr_decay(epoch):
        return 0.001 if epoch < 5 else 0.0001

    callbacks = [
        LearningRateScheduler(lr_decay),
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=callbacks
    )

    # Load the best checkpoint saved during training, rather than assuming
    # the final epoch's weights are the best ones.
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)


# --- 8. EVALUATE ---
loss, accuracy = model.evaluate(test_dataset, verbose=1)
print(f"\nTest/demo set loss: {loss:.4f}")
print(f"Test/demo set accuracy: {accuracy * 100:.2f}%")


# --- 9. SHOW TEST IMAGES: PREDICTED LABEL VS ACTUAL LABEL ---
def show_test_predictions(model, dataframe, number_to_show=12):
    sample = dataframe.sample(
        n=min(number_to_show, len(dataframe)),
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # Use the same preprocessing as the model's evaluation pipeline.
    images = []
    for path in sample["filename"]:
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = preprocess_input(image)
        images.append(image.numpy())

    probabilities = model.predict(np.stack(images), verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence_scores = np.max(probabilities, axis=1)

    columns = 4
    rows = math.ceil(len(sample) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (ax, (_, row)) in enumerate(zip(axes, sample.iterrows())):
        # Display the original image, not the normalized ResNet input.
        original = plt.imread(row["filename"])
        actual_label = row["label"]
        predicted_label = index_to_class[int(predicted_indices[i])]
        is_correct = predicted_label == actual_label

        ax.imshow(original)
        ax.set_title(
            f"Actual: {actual_label}\n"
            f"Predicted: {predicted_label} ({confidence_scores[i]:.1%})\n"
            f"{'CORRECT' if is_correct else 'INCORRECT'}",
            fontsize=9,
            color="green" if is_correct else "red"
        )
        ax.axis("off")

    fig.suptitle("Test Set Predictions: Actual vs Predicted", fontsize=16)
    plt.tight_layout()
    plt.show()


show_test_predictions(model, test_df, NUM_TEST_PREDICTIONS_TO_SHOW)

print("\nFinished. The saved model can be reused on future runs.")
