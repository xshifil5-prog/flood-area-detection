import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from io import BytesIO
# --------------------- Config ---------------------
st.set_page_config(page_title="Flood Segmentation Dashboard", layout="centered")

# ------------------ Model Loading ------------------
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load("best_unet_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------ Image Preprocessing ------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(output_tensor):
    mask = output_tensor[0][0].detach().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(binary_mask)

# ------------------ Metrics ------------------
def dice_coef(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def iou_coef(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    acc = (preds == targets).float().mean()
    return acc.item()

# ------------------ Streamlit UI ------------------
st.title("Flood Area Segmentation using U-Net + ResNet34")
st.write("Upload a satellite image and optionally a ground truth mask to segment and evaluate flood-affected regions.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
uploaded_mask = st.file_uploader("Upload Ground Truth Mask (optional)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Segment"):
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)[0][0]
            output_sigmoid = torch.sigmoid(output).numpy()
            binary_mask = (output_sigmoid > 0.5).astype(np.uint8)

        # Calculate flood percentage
        flood_percentage = (binary_mask.sum() / binary_mask.size) * 100

        # Create a PIL image from binary mask
        mask_img = Image.fromarray(binary_mask * 255)

        # Display mask
        st.image(mask_img, caption="Predicted Segmentation Mask", use_column_width=True)

        # Display flood info
        st.markdown(f"### 🌊 Estimated Flood Coverage: **{flood_percentage:.2f}%**")

        if flood_percentage > 40:
            st.error("🚨 Region is flooded. **Evacuation should be done immediately!**")
        elif flood_percentage > 30:
            st.warning("⚠️ Partial flooding detected. **Stay alert and monitor the situation.**")
        else:
            st.success("✅ No major flooding detected.")

        # Prepare download
        from io import BytesIO
        buffer = BytesIO()
        mask_img.save(buffer, format="PNG")
        byte_data = buffer.getvalue()

        st.download_button(
            label="📥 Download Mask",
            data=byte_data,
            file_name="segmented_mask.png",
            mime="image/png"
        )

        # ---------- Metrics Evaluation ----------
        if uploaded_mask:
            gt_mask = Image.open(uploaded_mask).convert("L").resize((256, 256))
            gt_tensor = transforms.ToTensor()(gt_mask).unsqueeze(0)

            dice = dice_coef(output, gt_tensor)
            iou = iou_coef(output, gt_tensor)
            acc = accuracy(output, gt_tensor)

            st.success(f"📊 Dice: `{dice:.4f}` | IoU: `{iou:.4f}` | Accuracy: `{acc:.4f}`")
        else:
            st.info("Upload a ground truth mask to view evaluation metrics.")
