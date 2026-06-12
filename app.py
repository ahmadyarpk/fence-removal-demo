import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ▶ Change these two values to match your HF username and model repo name
HF_REPO_ID   = "your-hf-username/fence-removal"   # e.g. "ahmadyar/fence-removal"
HF_FILENAME  = "ckpt_epoch_150.pth"

IMAGE_SIZE = 256

# ──────────────────────────────────────────
# MODEL DEFINITIONS  (identical to training)
# ──────────────────────────────────────────
class FenceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        try:
            self.model = models.segmentation.deeplabv3_resnet101(
                weights=models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            )
        except Exception:
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


class LaMaInpaintNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, img, mask_prob):
        x = torch.cat([img * (1 - mask_prob), mask_prob], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        return self.decoder(x)


# ──────────────────────────────────────────
# LOAD WEIGHTS  (downloaded once at startup)
# ──────────────────────────────────────────
print(f"Downloading model weights from HF Hub: {HF_REPO_ID} ...")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

checkpoint = torch.load(model_path, map_location=device)

fence_detector = FenceDetector().to(device)
inpaint_net    = LaMaInpaintNet().to(device)

fence_detector.load_state_dict(checkpoint["fence_detector"])
inpaint_net.load_state_dict(checkpoint["inpaint_net"])

fence_detector.eval()
inpaint_net.eval()
print("Models ready.")

# ──────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────
transform_img = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

# ──────────────────────────────────────────
# INFERENCE FUNCTION
# ──────────────────────────────────────────
def remove_fence(image: np.ndarray):
    """
    Takes a numpy RGB image (from Gradio), returns three PIL images:
    resized input | predicted fence mask | inpainted result
    """
    if image is None:
        return None, None, None

    img_pil    = Image.fromarray(image).convert("RGB")
    img_tensor = transform_img(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits  = fence_detector(img_tensor)
        mask_prob    = torch.sigmoid(mask_logits)
        clean_tensor = inpaint_net(img_tensor, mask_prob)

    resized_input = to_pil(img_tensor.squeeze(0).cpu())
    mask_img      = to_pil(mask_prob.squeeze(0).cpu()).convert("RGB")
    clean_img     = to_pil(clean_tensor.squeeze(0).cpu())

    return resized_input, mask_img, clean_img

# ──────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────
css = """
#title { text-align: center; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1rem; }
"""

with gr.Blocks(title="Fence Removal Demo", css=css) as demo:
    gr.Markdown("# 🚧 Deep Learning Fence Removal", elem_id="title")
    gr.Markdown(
        "Upload a photo taken through a fence. The model detects the fence "
        "and inpaints the occluded regions using a DeepLabV3 + LaMa-inspired pipeline.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📷 Input Image", type="numpy")
            run_btn = gr.Button("🔍 Remove Fence", variant="primary", size="lg")

    with gr.Row():
        out_resized = gr.Image(label="Resized Input  (256 × 256)")
        out_mask    = gr.Image(label="Predicted Fence Mask")
        out_clean   = gr.Image(label="✅ Inpainted Output")

    run_btn.click(
        fn=remove_fence,
        inputs=input_image,
        outputs=[out_resized, out_mask, out_clean],
    )

    # ── Example images (put a few in an examples/ folder in the repo) ──
    example_files = []
    if os.path.isdir("examples"):
        example_files = [
            [os.path.join("examples", f)]
            for f in sorted(os.listdir("examples"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if example_files:
        gr.Examples(examples=example_files, inputs=input_image)

    gr.Markdown(
        "---\n"
        "**Model:** DeepLabV3-ResNet101 fence segmentor + LaMa-inspired inpainting network  \n"
        "**Trained for:** 150 epochs  |  **Input size:** 256 × 256"
    )

demo.launch()
