
# Plant Disease Classification Using Deep Learning

## Project Overview

This project aims to classify plant diseases from leaf images using Deep Learning and Transfer Learning techniques. The PlantVillage dataset was used to train an image classification model capable of identifying multiple plant diseases and healthy plant conditions.

The project applies image preprocessing, data augmentation, transfer learning with EfficientNetB0, model evaluation, and hyperparameter tuning to investigate the effectiveness of deep learning for agricultural disease detection.

---

## Dataset

The project uses the PlantVillage Dataset containing images of healthy and diseased plant leaves.

Characteristics:

- 38 plant disease classes
- RGB leaf images
- Multiple crop species
- Images grouped into class folders

Dataset Structure:

```text
PlantVillage/
└── plantvillage dataset/
    └── color/
        ├── Apple___Apple_scab
        ├── Apple___Black_rot
        ├── Corn___Common_rust
        ├── Tomato___Early_blight
        └── ...
```

---

## Libraries Used

```python
TensorFlow
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
```

---

## Data Preprocessing

The images were preprocessed using ImageDataGenerator with:

- Rescaling
- Rotation augmentation
- Zoom augmentation
- Shear transformation
- Horizontal flipping
- 80:20 training-validation split

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
```

---

## Model Architecture

The model uses EfficientNetB0 as a transfer learning backbone.

Architecture:

1. EfficientNetB0 (pretrained on ImageNet)
2. Global Average Pooling Layer
3. Dropout (0.3)
4. Dense Layer (256 neurons)
5. Dropout (0.3)
6. Softmax Output Layer

```python
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
```

---

## Training Configuration

```python
Optimizer: Adam
Loss Function: Categorical Crossentropy
Batch Size: 32
Epochs: 10
Image Size: 128 × 128
```

Callbacks used:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

---

## Training Results

### Accuracy Analysis

The model exhibited poor learning performance throughout training. Training accuracy fluctuated around 10%, while validation accuracy remained approximately 10.15%.

Since the accuracy values were close to random guessing, the model failed to learn meaningful patterns from the PlantVillage dataset. The similar performance between training and validation sets suggests underfitting rather than overfitting.

Possible causes include:

- Incorrect label mapping
- Data preprocessing issues
- Class imbalance
- Insufficient feature extraction
- Model configuration problems

---

### Accuracy Curve

The training and validation accuracy curves remained nearly flat throughout training.

Observations:

- Training accuracy ≈ 10%
- Validation accuracy ≈ 10.15%
- No significant learning trend observed
- Severe underfitting detected

---

### Loss Analysis

Both training and validation losses decreased only slightly during training.

Observations:

- Loss remained relatively high
- Training and validation losses followed similar patterns
- No signs of overfitting
- Model optimization was limited

The loss curves indicate stable but ineffective learning.

---

## Model Evaluation

### Validation Performance

```text
Validation Loss: 3.3428
Validation Accuracy: 0.1015
```

The low validation accuracy indicates that the model correctly classified only a small proportion of plant disease images. The model performed only slightly better than random guessing across the 38 classes.

---

## Classification Report Analysis

The classification report revealed extremely poor class-level performance.

Key findings:

- Most classes recorded precision, recall, and F1-scores close to zero.
- The model frequently predicted a single dominant class.
- Generalization across disease categories was poor.
- Overall accuracy remained approximately 10%.

These results indicate severe underfitting and poor discriminative ability.

---

## Confusion Matrix Analysis

The confusion matrix showed that the model struggled to distinguish between disease categories.

Key observations:

- Predictions were concentrated in a single class.
- Very few correct classifications appeared along the diagonal.
- Most classes were consistently misclassified.
- Disease discrimination was extremely poor.

The results confirm that the model collapsed towards predicting one dominant class rather than learning features for all disease categories.

---

## Hyperparameter Tuning

The following hyperparameter combinations were prepared for future experiments:

```python
learning_rates = [0.001, 0.0005, 0.0001]

batch_sizes = [16, 32]

dropouts = [0.2, 0.3, 0.5]
```

---

## Model Saving

The trained model was saved using:

```python
model.save("plant_disease_classifier.keras")
```

---

## Future Improvements

To improve model performance:

- Fine-tune EfficientNetB0 layers
- Increase training epochs
- Use larger image sizes (224×224)
- Address class imbalance
- Improve preprocessing pipeline
- Optimize hyperparameters
- Experiment with alternative architectures such as:
  - MobileNetV2
  - ResNet50
  - DenseNet121
  - EfficientNetB3

---

## Project Structure

```text
Plant-Disease-Classification-Project/
│
├── README.md
├── plants.ipynb
├── image.png
├── best_model.keras
├── plant_disease_classifier.keras
└── .gitignore
```

---

## Conclusion

This project explored plant disease classification using transfer learning and deep learning techniques. Although EfficientNetB0 and image augmentation were implemented, the model achieved only 10.15% validation accuracy and demonstrated significant underfitting. The findings highlight the importance of effective preprocessing, model fine-tuning, and hyperparameter optimization when developing deep learning solutions for plant disease detection. Future improvements are expected to enhance classification accuracy and generalization performance.