import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Generator model (same as used during training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Inverse transform for display
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5 # from [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    return tensor

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    model.load_state_dict(torch.load("monet_generator_final_2.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Streamlit UI
st.title("🎨 Photo to Monet Style Transfer")
st.write("Upload a photo and get a Monet-style version!")

uploaded_file = st.file_uploader("Upload a photo (jpg/png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Original Photo", use_column_width=True)

    with st.spinner("Generating Monet-style image..."):
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            fake_monet = model(input_tensor)

        # Postprocess
        output_image = denormalize(fake_monet.squeeze()).cpu()
        output_pil = transforms.ToPILImage()(output_image)

    st.image(output_pil, caption="🎨 Monet-style Image", use_column_width=True)
