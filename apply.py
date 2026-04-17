import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_architecture import CNN
import torchvision.transforms.functional as TF

# 1. Load model
model = CNN()
checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Transform (match MNIST exactly)
transform = transforms.ToTensor()

# 3. Load image
image_path = "3.png"
image = Image.open(image_path).convert("L")


# IMPORTANT: resize with correct interpolation
image = image.resize((28, 28), Image.NEAREST)

# 4. Convert to tensor
image = transform(image)  # [1, 28, 28], values [0,1]

# 🔥 FIX 1: Ensure correct polarity
# MNIST = white digit (high values) on black (low values)
# If your digit looks dark on light background → invert
if image.mean() > 0.5:
    image = 1 - image

# 🔥 FIX 2: Normalize like MNIST (this stabilizes predictions)
# image = (image - 0.1307) / 0.3081
# print(image.min().item(), image.max().item(), image.mean().item())
# 5. Show image
plt.imshow(image.squeeze(), cmap='gray')
plt.title("Input to Model (after preprocessing)")
plt.axis('off')
# plt.show()

# 6. Add batch dimension
image = image.unsqueeze(0)  # [1, 1, 28, 28]

# 7. Predict
with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1).item()

    probs = F.softmax(output, dim=1)
    confidence = probs.max().item()

print(f"Predicted digit: {pred}, Confidence: {confidence:.4f}")
print("\nProbabilities:")
for i in range(10):
    print(f"Digit {i}: {probs[0][i].item():.4f}")
