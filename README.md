
# Pet Image Classification with CNN (PyTorch)

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify images of pets (cats and dogs).  
It covers **data preprocessing, model training, evaluation, and prediction**.

The pipeline includes:
- Custom train/test transformations
- Dataset loading with `ImageFolder`
- Train/validation splitting
- CNN model definition
- Training & validation loops with loss tracking
- Model evaluation with classification metrics
- Prediction on custom images



## 📂 Project Structure
```

PetImages/               # Dataset folder (cats and dogs)
orange.jpeg              # Example test image
main.py                  # Main script containing all code
README.md                # Project documentation

````

---

## ⚙️ Requirements
Install the required libraries before running the code:

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn pillow
````

---

## 🖼 Dataset

The dataset should be structured as follows:

```
PetImages/
    Cat/
        image1.jpg
        image2.jpg
        ...
    Dog/
        image1.jpg
        image2.jpg
        ...
```

---

## 🚀 How to Run

Open the Notebook
Launch Jupyter Notebook and open the .ipynb file containing this code.

```
bash
jupyter notebook
Ensure Dataset Availability
Place the dataset in the PetImages/ folder, structured like:
````

```
markdown

PetImages/
    Cat/
        image1.jpg
        ...
    Dog/
        image1.jpg
        ...
````   


Run All Cells
In Jupyter:

Go to Kernel → Restart & Run All

Or press Shift + Enter to run each cell in sequence.

View Outputs

Training and validation loss plots will be displayed inline.

The classification report will be printed in the output cell.

Predictions for custom images will appear at the end.
---

## 🧠 Model Architecture

The **SimpleCNN** architecture:

* **Conv Layer 1:** 3 input channels → 32 filters (3x3 kernel, padding=1)
* **Max Pooling** (2x2)
* **Conv Layer 2:** 32 → 64 filters (3x3 kernel, padding=1)
* **Max Pooling** (2x2)
* **Fully Connected Layer 1:** Flattened → 128 units (ReLU)
* **Fully Connected Layer 2:** 128 → 2 output units (logits)

---

## 📊 Training Process

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (lr=0.001)
* **Epochs:** 20
* **Batch Size:** 32 for training/validation, 64 for testing
* **Transforms:**

  * Training: Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize
  * Testing: Resize, Normalize

The script plots **Training vs. Validation Loss** over epochs.

---

## 📈 Evaluation

The model is evaluated using:

* **Classification Report:** Precision, Recall, F1-Score per class
* **Softmax Probabilities:** For predictions

Example output:

```
              precision    recall  f1-score   support

        Cat       0.95      0.93      0.94       500
        Dog       0.92      0.94      0.93       500

    accuracy                           0.94      1000
   macro avg       0.94      0.94      0.94      1000
weighted avg       0.94      0.94      0.94      1000
```

---

## 🖼 Custom Image Prediction

You can test the trained model on your own image:

```python
img_path = "orange.jpeg"  # Path to your image
```

Output includes:

* **Predicted Label**
* **Class Probabilities**

Example:

```
Predicted label: Cat
Probabilities: [[0.85, 0.15]]
```

---

## 📌 Notes

* Ensure the dataset is balanced for better accuracy.
* Adjust transforms for better generalization.
* For GPU acceleration, ensure CUDA is available.
* Model hyperparameters (learning rate, batch size, etc.) can be tuned for better performance.

---

## 📜 License

This project is licensed under the MIT License.



