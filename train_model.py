import os
import shutil
import kagglehub
import tensorflow as tf
import openpyxl
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# --- CONFIG ---
IMG_SIZE = 256
BATCH_SIZE = 32
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random.seed(42)  # For reproducibility

# --- DOWNLOAD DATASET ---
root = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification") + "/dataset/"

# --- SPLIT DATASET ---
class_names = [
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d))
    and not d.startswith('.')
    and d.lower() != "dataset"
]

split_data_dir = os.path.join(os.getcwd(), "dataset_split")
for split in ["train", "val", "test"]:
    for class_name in class_names:
        os.makedirs(os.path.join(split_data_dir, split, class_name), exist_ok=True)

def copy_images(img_list, class_name, class_path, split_name, split_data_dir):
    for img in img_list:
        src = os.path.join(class_path, img)
        dst = os.path.join(split_data_dir, split_name, class_name, img)
        if os.path.isfile(src):  # Only copy files
            shutil.copy(src, dst)

for class_name in class_names:
    class_path = os.path.join(root, class_name)
    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))
              and "_mask" not in img.lower()]  # Ignore mask/mask-like files
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    copy_images(train_imgs, class_name, class_path, "train", split_data_dir)
    copy_images(val_imgs, class_name, class_path, "val", split_data_dir)
    copy_images(test_imgs, class_name, class_path, "test", split_data_dir)

# --- CLEANUP: ENSURE NO NON-CLASS FOLDERS IN SPLIT DIRS ---
for split in ["train", "val", "test"]:
    split_path = os.path.join(split_data_dir, split)
    for item in os.listdir(split_path):
        item_path = os.path.join(split_path, item)
        if not os.path.isdir(item_path):
            os.remove(item_path)
        elif item.lower() == "dataset":
            shutil.rmtree(item_path)

print('Dataset split complete!\n')

for split in ["train", "val", "test"]:
    split_count = 0
    for class_name in class_names:
        split_count += len(os.listdir(os.path.join(split_data_dir, split, class_name)))
    print(f"Number of images in {split} split: {split_count}")

# --- LOAD DATASETS ---
train_dir = os.path.join(split_data_dir, "train")
val_dir = os.path.join(split_data_dir, "val")
test_dir = os.path.join(split_data_dir, "test")

# Show actual folders in each split to verify
for split, path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
    print(f"Folders in {split} split: {os.listdir(path)}\n")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    color_mode='rgb'
)
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    color_mode='rgb'
)
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode='rgb'
)

class_names = train_data.class_names
num_classes = len(class_names)


# --- QUICK VISUAL CHECK OF A FEW RANDOM IMAGES ---
def show_random_image_per_category(dataset, class_names):
    # Collect all images by class
    images_by_class = {i: [] for i in range(len(class_names))}
    for img, lbl in dataset.unbatch():
        lbl_idx = np.argmax(lbl.numpy())
        images_by_class[lbl_idx].append(img.numpy())
    
    plt.figure(figsize=(4 * len(class_names), 4))
    for class_idx in range(len(class_names)):
        # Randomly select an image for this class
        imgs = images_by_class[class_idx]
        if imgs:
            img = imgs[np.random.randint(len(imgs))]
            plt.subplot(1, len(class_names), class_idx + 1)
            plt.imshow(img.astype(np.uint8))
            plt.title(class_names[class_idx])
            plt.axis('off')
    plt.tight_layout()
    plt.show()

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

print("\nTrain Data Samples")
show_random_image_per_category(train_data, class_names)

print("\nTest Data Samples")
show_random_image_per_category(test_data, class_names)

print("\nValidation Data Samples")
show_random_image_per_category(val_data, class_names)

print("\n")

# --- PREFETCHING ---
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# --- SANITY TEST: MODEL SHOULD OVERFIT TINY BATCH ---
def do_sanity_test(train_data, class_names):
    print("Sanity Test")
    wanted_classes = [0, 1]
    max_per_class = 10
    images = []
    labels = []
    class_counts = {c: 0 for c in wanted_classes}
    for img, lbl in train_data.unbatch():
        label = np.argmax(lbl.numpy())
        if label in wanted_classes and class_counts[label] < max_per_class:
            images.append(img.numpy())
            labels.append(label)
            class_counts[label] += 1
        if all(class_counts[c] == max_per_class for c in wanted_classes):
            break
    images = np.stack(images)
    labels = np.array(labels)
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(2, activation='softmax')(x)
    sanity_model = tf.keras.Model(inp, out)
    sanity_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    relabel = {c: i for i, c in enumerate(wanted_classes)}
    labels_re = np.array([relabel[l] for l in labels])
    labels_re_cat = tf.keras.utils.to_categorical(labels_re, num_classes=2)
    sanity_history = sanity_model.fit(
        images, labels_re_cat, epochs=30, batch_size=4, verbose=0
    )
    final_acc = sanity_history.history["accuracy"][-1]
    print(f"Sanity test final train accuracy: {final_acc:.4f}")
    if final_acc <= 0.95:
        raise RuntimeError("Sanity test FAILED: Model did NOT overfit, check label/image pipeline!")

# do_sanity_test(train_data, class_names)

# --- DATA AUGMENTATION PIPELINE ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(0.2),
])

# --- TRANSFER LEARNING MODEL (EFFICIENTNETB0) ---
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base for initial training

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- INITIAL TRAINING: FROZEN BASE ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_eye_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# --- FINE-TUNE THE LAST 25 LAYERS, ADD DATA AUGMENTATION ---
for layer in base_model.layers[:-25]:
    layer.trainable = False
for layer in base_model.layers[-25:]:
    layer.trainable = True

inputs_aug = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
aug = data_augmentation(inputs_aug)
aug = tf.keras.applications.efficientnet.preprocess_input(aug)
x = base_model(aug, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model_aug = tf.keras.Model(inputs_aug, outputs)
model_aug.set_weights(model.get_weights())  # Transfer weights

model_aug.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history_fine = model_aug.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# --- EVALUATION: FINAL MODEL (model_aug) ---
final_model = model_aug
final_loss, final_acc = final_model.evaluate(test_data, verbose=1)

# --- EVALUATION: BEST MODEL ---
best_model = tf.keras.models.load_model("best_eye_model.keras")
best_loss, best_acc = best_model.evaluate(test_data, verbose=1)

# --- CONFUSION MATRIX & REPORT: BASED ON BEST MODEL ---
y_true = []
y_pred = []

for images, labels in test_data:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

# --- SAVE CONFUSION MATRIX IMAGE ---
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_type = "EfficientNetB0"
filename = f"confusion_matrix_{model_type}_img{IMG_SIZE}_bs{BATCH_SIZE}_ep{len(history.epoch)+len(history_fine.epoch)}_{now}_eye.png"

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Best Model)")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(filename, dpi=300)
plt.close()
print(f"Confusion matrix saved as {filename}")

report_str = classification_report(y_true, y_pred, target_names=class_names)

# --- SAVE RESULTS TO EXCEL ---
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results"

ws.append(["Metric", "Final Model", "Best Model"])
ws.append(["Test Accuracy", final_acc, best_acc])
ws.append(["Test Loss", final_loss, best_loss])
ws.append(["Epochs (Total)", len(history.epoch) + len(history_fine.epoch), len(history.epoch) + len(history_fine.epoch)])
ws.append(["Confusion Matrix Image (Best Model)", filename, filename])

ws.append([])
ws.append(["Model Summary (Best Model)", ""])
model_summary = []
best_model.summary(print_fn=lambda x: model_summary.append(x))
for line in model_summary:
    ws.append([line])

ws.append([])
ws.append(["Confusion Matrix (Best Model)"])
ws.append([""] + class_names)
for i, row in enumerate(cm):
    ws.append([class_names[i]] + row.tolist())

ws.append([])
ws.append(["Classification Report (Best Model)"])
for line in report_str.strip().split("\n"):
    ws.append([line])

wb.save("eye_classification_results.xlsx")
print("Results saved to eye_classification_results.xlsx")