import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load label names
CLASS_NAMES_FILE = './nabirds/classes.txt'  # optional
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
else:
    class_names = [f"Species {i}" for i in range(555)]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 555)
model.load_state_dict(torch.load('nabirds_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Open camera
cap = cv2.VideoCapture(0)
print("[INFO] Starting bird classification. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        bird_name = class_names[pred_idx]

    # Display
    label_text = f"{bird_name}"
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('BirdCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
