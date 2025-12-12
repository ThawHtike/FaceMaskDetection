# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os








# --- 1. SET CONFIGURATION PARAMETERS (Hyperparameters) ---
# Set initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-4          # Starting learning rate
EPOCHS = 20             # How many times to loop over the whole dataset
BS = 32                 # How many images to process at a time
DIRECTORY = "dataset"   # Folder containing the 'with_mask' and 'without_mask' subfolders
MODEL_PATH = "mask_detector.model" # Where to save the final trained model

# --- 2. LOAD DATA AND PREPARE LABELS ---
print("[INFO] Loading images from disk...")
data = []
labels = []

# Loop over all image paths in the dataset directory
for imagePath in list(paths.list_images(DIRECTORY)):
    # Extract the class label from the directory name (e.g., 'with_mask' or 'without_mask')
    label = imagePath.split(os.path.sep)[-2]

    # Load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224)) # Resize to 224x224 for MobileNetV2
    image = img_to_array(image)
    image = preprocess_input(image) # MobileNetV2 requires specific preprocessing

    # Update the data and labels lists
    data.append(image)
    labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels (convert "with_mask" to [1, 0], "without_mask" to [0, 1])
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels) # Import 'to_categorical' from tensorflow.keras.utils

# Split the data into training (80%) and testing (20%) sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# --- 3. DATA AUGMENTATION ---
# Construct the training image generator for data augmentation
# This helps the model generalize better by creating slightly modified copies of images (e.g., small rotations, flips)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# --- 4. BUILD THE MOBILE-NETV2 MODEL ---
# Load MobileNetV2 pre-trained on ImageNet (but don't train the top layers yet)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2 outputs: Mask or No Mask

# Place the head model on top of the base model (This is our final model)
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers in the base model so they are not updated during the first training step
for layer in baseModel.layers:
    layer.trainable = False

# --- 5. COMPILE AND TRAIN THE MODEL ---
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) # Use the Adam optimizer
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network
print("[INFO] Training head network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# --- 6. MAKE PREDICTIONS AND EVALUATE ---
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# For each image in the testing set, we need to find the index of the label with the largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# --- 7. SAVE THE MODEL TO DISK ---
print("[INFO] Saving mask detector model...")
model.save(MODEL_PATH, save_format="h5")

# --- 8. PLOT TRAINING LOSS AND ACCURACY ---
# This part is optional but useful for your Master's report
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")