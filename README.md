# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
The experiment aims to develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron. The model will be trained on a dataset containing images of various defected and non-defected capacitors to enhance defect detection accuracy. Optimization techniques will be applied to improve performance, and the model will be evaluated to ensure reliable classification for capacitor quality assessment in manufacturing.
## DESIGN STEPS
### STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.
</br>

### STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.
</br>

### STEP 3:
Train the model using the training dataset with forward and backward propagation.
<br/>

### STEP 4:
Train the model using the training dataset with forward and backward propagation.
<br/>

### STEP 5:
Make predictions on new data using the trained model.
<br/>

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT

# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features,1)


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float() )
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2025-04-16 183953](https://github.com/user-attachments/assets/b51dd9ba-e8d3-4d1d-9948-c787076e61d6)


### Confusion Matrix

![Screenshot 2025-04-16 185609](https://github.com/user-attachments/assets/8ea8b34c-5f06-4717-824b-d9bc6ca5779f)

### Classification Report

![Screenshot 2025-04-16 185614](https://github.com/user-attachments/assets/47e0524d-ee69-4eea-add3-5c4db55d3320)


### New Sample Prediction

![Screenshot 2025-04-16 185648](https://github.com/user-attachments/assets/0ad3c921-acc5-47a2-ae10-e19c2c2fa1f3)

## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
