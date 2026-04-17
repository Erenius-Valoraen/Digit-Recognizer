import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_architecture import CNN
import torchvision.transforms.functional as TF


model = CNN()
checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


transform = transforms.ToTensor()


image_path = "3.png"
image = Image.open(image_path).convert("L")



image = image.resize((28, 28), Image.NEAREST)


image = transform(image)


if image.mean() > 0.5:
    image = 1 - image


plt.imshow(image.squeeze(), cmap='gray')
plt.title("Input to Model (after preprocessing)")
plt.axis('off')
# plt.show()


image = image.unsqueeze(0)  # [1, 1, 28, 28]


with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1).item()

    probs = F.softmax(output, dim=1)
    confidence = probs.max().item()

print(f"Predicted digit: {pred}, Confidence: {confidence:.4f}")
print("\nProbabilities:")
for i in range(10):
    print(f"Digit {i}: {probs[0][i].item():.4f}")
