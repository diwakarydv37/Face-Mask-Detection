# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


INIT_LR = 1e-5
EPOCHS =20
BS = 30
.5

DIRECTORY = r"C:\Users\RAKESH\PycharmProjects\RakeshProject\mlpackage\project college\dataset"
CATEGORIES = ["with_mask", "without_mask"]


print(" images...")

a= []
labels = []

for section in CATEGORIES:
    place = os.path.join(DIRECTORY, section)
    for img in os.listdir(place):
    	img_place = os.path.join(place, img)
    	name = load_img(img_place, target_size=(220, 220))
    	name = img_to_array(name)
    	name = preprocess_input(name)

    	a.append(name)
    	labels.append(section)


lab = LabelBinarizer()
labels = lab.fit_transform(labels)
labels = to_categorical(labels)

a = np.array(a, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(a, labels,
	test_size=0.33, stratify=labels, random_state=21)


totl= ImageDataGenerator(

	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
    rescale=None)


primaryMode = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(220, 220, 3)))


SecondaryMode = primaryMode.output
SecondaryMode = AveragePooling2D(pool_size=(6, 6))(SecondaryMode)
SecondaryMode = Flatten(name="flatten")(SecondaryMode)
SecondaryMode = Dense(120, activation="relu")(SecondaryMode)
SecondaryMode = Dropout(0.4)(SecondaryMode)
SecondaryMode = Dense(2, activation="softmax")(SecondaryMode)


model = Model(inputs=primaryMode.input, outputs=SecondaryMode)


for layer in primaryMode.layers:
	layer.trainable = False


print("compiling ...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print(" training ...")
H = model.fit(
	totl.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


print(" evaluating ...")
Z_pred = model.predict(testX, batch_size=BS)


Z_pred = np.argmax(Z_pred, axis=1)


print(classification_report(testY.argmax(axis=1), Z_pred,
	target_names=lab.classes_))


print(" saving mask detector ...")
model.save("mask_detector.model", save_format="h5")


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