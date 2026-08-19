# ==========================================================
# PART A - CNN Image Classification using MNIST
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import warnings
warnings.filterwarnings("ignore")   


def train_model():
    # -------------------------
    # Load Dataset
    # -------------------------
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("=" * 45)
    print("MNIST DATASET INFORMATION")
    print("=" * 45)
    print("Training Images :", x_train.shape)
    print("Training Labels :", y_train.shape)
    print("Testing Images  :", x_test.shape)
    print("Testing Labels  :", y_test.shape)
    print("Image Size      :", x_train[0].shape)
    print("Classes         :", np.unique(y_train))

    class_names = [str(i) for i in range(10)]
    print("Class Names :", class_names)


    # -------------------------
    # Display Sample Images
    # -------------------------
    plt.figure(figsize=(10,2))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(y_train[i])
        plt.axis("off")
    plt.suptitle("Sample MNIST Images")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Preprocessing
    # -------------------------
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # -------------------------
    # Build CNN Model
    # -------------------------
    model = Sequential([
        Input(shape=(28,28,1)),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.summary()

    # -------------------------
    # Compile Model
    # -------------------------
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------
    # Train Model
    # -------------------------
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )


    model.save("mnist_cnn_model.keras")
    print("Model saved successfully!")


    # -------------------------
    # Evaluate Model
    # -------------------------
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print("\nTest Loss :", round(test_loss,4))
    print("Test Accuracy :", round(test_accuracy*100,2), "%")

    # -------------------------
    # Accuracy & Loss Graph
    # -------------------------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Predictions
    # -------------------------
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # -------------------------
    # Actual vs Predicted
    # -------------------------
    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap="gray")
        plt.title(f"A:{y_test[i]} P:{predicted_labels[i]}")
        plt.axis("off")

    plt.suptitle("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Correct Predictions
    # -------------------------
    correct = np.where(predicted_labels == y_test)[0]

    plt.figure(figsize=(10,4))

    for i, index in enumerate(correct[:10]):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[index].reshape(28,28), cmap="gray")
        plt.title(predicted_labels[index])
        plt.axis("off")

    plt.suptitle("Correct Predictions")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Incorrect Predictions
    # -------------------------
    incorrect = np.where(predicted_labels != y_test)[0]

    plt.figure(figsize=(10,4))

    for i, index in enumerate(incorrect[:10]):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[index].reshape(28,28), cmap="gray")
        plt.title(f"A:{y_test[index]} P:{predicted_labels[index]}")
        plt.axis("off")

    plt.suptitle("Incorrect Predictions")
    plt.tight_layout()
    plt.show()

    print("\nCNN Image Classification Completed Successfully!")
    print("Final Test Accuracy :", round(test_accuracy*100,2), "%")

    print("\nConclusion:")
    print("The CNN model classified handwritten digits with high accuracy.")
    print("The model achieved", round(test_accuracy * 100, 2), "% accuracy on the test dataset.")






# ========================================================================
# PART B - Streamlit Web Application for MNIST Digit Classification
# ========================================================================

import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

def streamlit_app():

    st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

    st.title("Handwritten Digit Recognition using CNN")
    st.write("Upload an image of a handwritten digit (0-9).")

    # Load trained model
    model = load_model("mnist_cnn_model.keras")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(image, caption="Uploaded Image", width=200)

        # Preprocess image
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = image.point(lambda x: 255 if x > 100 else 0)

        img = np.array(image).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        if st.button("Predict"):

            prediction = model.predict(img, verbose=0)

            digit = np.argmax(prediction)

            confidence = np.max(prediction) * 100

            st.success(f"Predicted Digit : {digit}")
            st.info(f"Confidence : {confidence:.2f}%")




import sys

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        streamlit_app()