import os
import pandas as pd
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
import torch

# Set environment variable for HF hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Model Configuration
model_id = "THUDM/CogVideoX-5b"
model_string = "cogvideox-5b"

# Load models
transformer = CogVideoXTransformer3DModel.from_pretrained(
    f"camenduru/{model_string}-float16", subfolder="transformer", torch_dtype=torch.float16
)
text_encoder = T5EncoderModel.from_pretrained(
    f"camenduru/{model_string}-float16", subfolder="text_encoder", torch_dtype=torch.float16
)
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float16
)

# Create pipeline
pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_sequential_cpu_offload()

csv_file = os.path.join(os.getcwd(), "video_captions.csv")
data = pd.read_csv(csv_file)

# Directory to save results
output_dir = f"results_{model_string}"
os.makedirs(output_dir, exist_ok=True)

# Generate videos for each caption
for index, row in data.iterrows():
    filename = row["filename"]
    prompt = row["caption"]
    
    print(f"\n\nGenerating a synthetic video for: {filename}")
    
    # Generate video 18-12
    video = pipe(prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=12, num_frames=24).frames[0]
    
    # Save video
    output_path = os.path.join(output_dir, f"{filename}_Fake.mp4")
    export_to_video(video, output_path, fps=3)

    print(f"Video saved: {output_path}")
