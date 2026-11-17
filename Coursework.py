
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow.keras.preprocessing import image

# ==========================================
# 1. DEFINE THE DATASET DIRECTORY
# ==========================================

dataset_path = r"C:\Users\divin\Documents\Intel-Image-Classification"

train_dir = os.path.join(dataset_path, "seg_train")
test_dir = os.path.join(dataset_path, "seg_test")

print("Training directory:", train_dir)
print("Testing directory:", test_dir)

# Check whether directories exist
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")

if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Testing directory not found: {test_dir}")


# ==========================================
# 2. EXPLORE THE DATASET
# ==========================================

classes = sorted([
    folder for folder in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, folder))
    and not folder.startswith(".")
])

print("\nNumber of classes:", len(classes))
print("Image classes:", classes)

# if len(classes) < 3:
#     raise ValueError("The dataset must contain at least three classes.")

# Count images in each class
class_counts = {}

for class_name in classes:

    class_path = os.path.join(train_dir, class_name)

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    class_counts[class_name] = len(image_files)

# Create DataFrame
df = pd.DataFrame(
    list(class_counts.items()),
    columns=["Class", "Number of Images"]
)

print("\nImage distribution:")
print(df)

print("\nTotal training images:", df["Number of Images"].sum())


# ==========================================
# 3. VISUALISE CLASS DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=df,
    x="Class",
    y="Number of Images"
)

plt.title("Distribution of Images Across Classes")
plt.xlabel("Image Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ==========================================
# 4. DISPLAY SAMPLE IMAGES
# ==========================================

plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes):

    class_path = os.path.join(train_dir, class_name)

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_files:
        continue

    img_name = image_files[0]
    img_path = os.path.join(class_path, img_name)

    img = image.load_img(img_path)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis("off")

plt.suptitle("Sample Images from Each Class")
plt.tight_layout()
plt.show()


# ==========================================
# 5. RESIZE AND LOAD IMAGES
# ==========================================

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ==========================================
# 6. NORMALISE IMAGE PIXEL VALUES
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# 7. OPTIMISE DATASET PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 8. VERIFY DATA PREPROCESSING
# ==========================================

for images, labels in train_dataset.take(1):

    print("\n========== PREPROCESSING RESULTS ==========")

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    print(
        "Minimum pixel value:",
        tf.reduce_min(images).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(images).numpy()
    )

    print("Number of classes:", len(classes))

    print("===========================================")


# ==========================================
# 9. DISPLAY NORMALISED IMAGE
# ==========================================

class_names = train_dataset.class_names if hasattr(
    train_dataset, "class_names"
) else classes

for images, labels in train_dataset.take(1):

    plt.figure(figsize=(10, 8))

    for i in range(min(9, len(images))):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy())

        label_index = np.argmax(labels[i].numpy())

        plt.title(class_names[label_index])

        plt.axis("off")

    plt.suptitle("Resized and Normalised Images")
    plt.tight_layout()
    plt.show()

# Identify largest and smallest classes
largest_class = df.loc[
    df["Number of Images"].idxmax()
]

smallest_class = df.loc[
    df["Number of Images"].idxmin()
]

imbalance_ratio = (
    largest_class["Number of Images"] /
    smallest_class["Number of Images"]
)

print("\nCLASS IMBALANCE ANALYSIS")

print(
    "Largest class:",
    largest_class["Class"],
    "-",
    largest_class["Number of Images"]
)

print(
    "Smallest class:",
    smallest_class["Class"],
    "-",
    smallest_class["Number of Images"]
)

print("Imbalance ratio:", round(imbalance_ratio, 2))

if imbalance_ratio < 1.5:
    print("The dataset is relatively balanced.")
elif imbalance_ratio < 2:
    print("The dataset has some class imbalance.")
else:
    print("The dataset has significant class imbalance.")

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1)

])

# Augmentation is applied only to training images
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

print("\nAdditional preprocessing completed successfully.")

for images, labels in train_dataset.take(1):

    print("\nPREPROCESSING VERIFICATION")

    print("Image shape:", images.shape)

    print("Label shape:", labels.shape)

    print(
        "Minimum pixel value:",
        tf.reduce_min(images).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(images).numpy()
    )