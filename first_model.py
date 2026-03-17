import numpy as np
import torch                    # The AI engine
import torch.nn as nn           # nn = neural network tools
import matplotlib.pyplot as plt

# ------------------------------------------
# STEP 1: Create our training data
# ------------------------------------------

# We will generate 100 images as numpy arrays
# 50 will be orange (label 0)
# 50 will be blue (label 1)

# Each image is 32x32 pixels with 3 channels (RGB)
# So each image is a (32, 32, 3) array

images = []   # This list will hold all our images
labels = []   # This list will hold the correct answer for each image

# Create 50 ORANGE images
for i in range(50):
    # np.ones creates an array filled with 1s
    # Multiply by [255, 165, 0] to make every pixel orange
    img = np.ones((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 255   # Red channel high
    img[:, :, 1] = 165   # Green channel medium
    img[:, :, 2] = 0     # Blue channel zero

    # Add small random noise so images aren't identical
    # This makes the model learn better
    img += np.random.randn(32, 32, 3) * 10

    images.append(img)
    labels.append(0)      # 0 = orange

# Create 50 BLUE images
for i in range(50):
    img = np.ones((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 0     # Red channel zero
    img[:, :, 1] = 0     # Green channel zero
    img[:, :, 2] = 255   # Blue channel high

    img += np.random.randn(32, 32, 3) * 10

    images.append(img)
    labels.append(1)      # 1 = blue

# Convert our lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

print("Images shape:", images.shape)
# Expected: (100, 32, 32, 3)
# 100 images, each 32x32 pixels, 3 color channels

print("Labels shape:", labels.shape)
# Expected: (100,) - one label per image

print("First 5 labels:", labels[:5])
# Expected: [0, 0, 0, 0, 0] - first 5 are orange

# ------------------------------------------
# STEP 2: Prepare data for PyTorch
# ------------------------------------------

# PyTorch expects images in a specific format
# It wants: (batch, channels, height, width)
# We currently have: (batch, height, width, channels)
# So we need to swap the axes using transpose

images = images.transpose(0, 3, 1, 2)
# 0 = keep batch dimension first
# 3 = move channels (was last) to second position
# 1 = height goes third
# 2 = width goes last

print("Images shape after transpose:", images.shape)
# Expected: (100, 3, 32, 32)

# Normalize pixel values from 0-255 to 0-1
# Neural networks learn better with small numbers
images = images / 255.0

# Convert numpy arrays to PyTorch tensors
# A tensor is PyTorch's version of a numpy array
X = torch.tensor(images, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------------------
# STEP 3: Build the Neural Network
# ------------------------------------------

# nn.Module is the base class for all neural networks in PyTorch
# We are creating our own class that inherits from it
class ImageClassifier(nn.Module):

    # __init__ runs when we create the model
    # This is where we define the layers
    def __init__(self):
        super().__init__()   # Always call this first - sets up PyTorch internals

        # Flatten converts our 3D image (3, 32, 32) into a 1D list of numbers
        # 3 x 32 x 32 = 3072 numbers
        self.flatten = nn.Flatten()

        # nn.Sequential runs layers one after another like a pipeline
        self.layers = nn.Sequential(

            # First layer: takes 3072 inputs, outputs 128 values
            # Think of it as: 3072 pixel values go in, 128 pattern detectors come out
            nn.Linear(3072, 128),

            # ReLU is an activation function
            # It introduces non-linearity - helps the model learn complex patterns
            # Simple rule: if number is negative make it 0, otherwise keep it
            nn.ReLU(),

            # Second layer: takes 128 inputs, outputs 64 values
            nn.Linear(128, 64),

            nn.ReLU(),

            # Final layer: takes 64 inputs, outputs 2 values
            # 2 outputs because we have 2 classes: orange(0) and blue(1)
            nn.Linear(64, 2)
        )

    # forward defines what happens when data passes through the model
    def forward(self, x):
        x = self.flatten(x)    # Flatten image to 1D
        x = self.layers(x)     # Pass through all layers
        return x               # Return the result

# Create an instance of our model
model = ImageClassifier()

# Print the model structure so we can see all layers
print(model)

# ------------------------------------------
# STEP 4: Define how the model learns
# ------------------------------------------

# Loss function measures how wrong the model's guess is
# CrossEntropyLoss is standard for classification problems
loss_function = nn.CrossEntropyLoss()

# Optimizer decides how to adjust the model based on the loss
# Adam is the most popular optimizer - it's smart about adjusting
# lr = learning rate - how big of a step to take when adjusting
# Small lr = learns slowly but carefully
# Large lr = learns fast but might overshoot
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel is ready to train!")

# ------------------------------------------
# STEP 5: Train the model
# ------------------------------------------

# An epoch is one full pass through ALL the training data
# We will do 20 passes so the model sees the data 20 times
epochs = 20

# This list will store the loss at each epoch so we can plot it later
loss_history = []

print("Starting training...\n")

# Loop through the data 20 times
for epoch in range(epochs):

    # Forward pass - send images through the model and get predictions
    # X contains all 100 images
    predictions = model(X)

    # Calculate how wrong the predictions are
    # predictions = what the model guessed
    # y = the correct answers
    loss = loss_function(predictions, y)

    # Backward pass - this is where learning happens
    # Zero out old gradients first (PyTorch accumulates them by default)
    optimizer.zero_grad()

    # Calculate gradients - figures out which direction to adjust
    loss.backward()

    # Update the model weights based on gradients
    optimizer.step()

    # Record the loss value for this epoch
    # .item() converts a PyTorch tensor to a regular Python number
    loss_history.append(loss.item())

    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/20 - Loss: {loss.item():.4f}")

print("\nTraining complete!")

# ------------------------------------------
# STEP 6: Plot the loss over time
# ------------------------------------------

# A dropping loss means the model is learning!
plt.plot(loss_history)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ------------------------------------------
# STEP 7: Test the model
# ------------------------------------------

# torch.no_grad() tells PyTorch we are not training
# so it doesn't waste memory tracking gradients
with torch.no_grad():

    # Get predictions for all images
    outputs = model(X)

    # torch.argmax finds the index of the highest value
    # If output is [0.2, 0.9] → argmax = 1 → predicted blue
    # If output is [0.8, 0.1] → argmax = 0 → predicted orange
    predicted_labels = torch.argmax(outputs, dim=1)

    # Compare predicted labels to correct labels
    correct = (predicted_labels == y).sum().item()
    total = len(y)
    accuracy = correct / total * 100

    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")