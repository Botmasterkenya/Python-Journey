# We are importing all our installed libraries to confirm they work
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# Print each library version so we can confirm they loaded
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# This checks if your computer can use GPU acceleration
# Since we installed CPU version, this will say False - that's totally fine
print("GPU available:", torch.cuda.is_available())

print("\nAll libraries loaded successfully! You are ready to build AI models!")
