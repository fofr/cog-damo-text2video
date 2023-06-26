import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from cog import BasePredictor, Input, Path
import imageio

MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe_xl = DiffusionPipeline.from_pretrained(
            MODEL_CACHE + "/xl",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        self.pipe_576w = DiffusionPipeline.from_pretrained(
            MODEL_CACHE + "/576w",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        for p in [self.pipe_576w, self.pipe_xl]:
            p.scheduler = DPMSolverMultistepScheduler.from_config(
                p.scheduler.config
            )
            p.enable_model_cpu_offload()
            p.enable_vae_slicing()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="An astronaut riding a horse"
        ),
        negative_prompt: str = Input(
            description="Negative prompt", default=None
        ),
        num_frames: int = Input(
            description="Number of frames for the output video", default=24
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        width: int = Input(
            description="Width of the output video", ge=256, default=576
        ),
        height: int = Input(
            description="Height of the output video", ge=256, default=320
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=1.0, le=100.0, default=7.5
        ),
        fps: int = Input(description="fps for the output video", default=8),
        fast: bool = Input(description="fast => 576w, normal = xl", default=True),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        if fast:
            pipe = self.pipe_576w
        else:
            pipe = self.pipe_xl

        frames = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames

        out = "/tmp/out.mp4"
        writer = imageio.get_writer(out, format="FFMPEG", fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return Path(out)
