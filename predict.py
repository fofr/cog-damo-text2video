import os
from typing import List
from cog import BasePredictor, Input, Path
import inference
import shutil

MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass
        # self.pipe_xl = DiffusionPipeline.from_pretrained(
        #     MODEL_CACHE + "/xl",
        #     torch_dtype=torch.float16,
        #     local_files_only=True,
        # ).to("cuda")

        # self.pipe_576w = DiffusionPipeline.from_pretrained(
        #     MODEL_CACHE + "/576w",
        #     torch_dtype=torch.float16,
        #     local_files_only=True,
        # ).to("cuda")

        # for p in [self.pipe_576w, self.pipe_xl]:
        #     p.scheduler = DPMSolverMultistepScheduler.from_config(
        #         p.scheduler.config
        #     )
        #     p.enable_model_cpu_offload()
        #     p.enable_vae_slicing()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="An astronaut riding a horse"
        ),
        negative_prompt: str = Input(
            description="Negative prompt", default=None
        ),
        init_video: Path = Input(
            description="URL of the initial video (optional)", default=None
        ),
        init_weight: float = Input(
            description="Strength of init_video", default=0.5
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
        model: str = Input(
            description="Model to use", default="xl", choices=["xl", "576w", "potat1"]
        ),
        batch_size: int = Input(description="Batch size", default=1, ge=1),
        remove_watermark: bool = Input(
            description="Remove watermark", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        shutil.rmtree("output", ignore_errors=True)
        os.makedirs("output", exist_ok=True)

        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
            "num_frames": num_frames,
            "num_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "fps": fps,
            "device": "cuda",
            "output_dir": "output",
            "remove_watermark": remove_watermark,
        }

        args['model'] = MODEL_CACHE + "/" + model

        if init_video is not None:
            # for some reason I need to copy the file to make it work
            if os.path.exists("input.mp4"):
                os.unlink("input.mp4")
            shutil.copy(init_video, "input.mp4")

            args["init_video"] = "input.mp4"
            args["init_weight"] = init_weight
            print("init_video", os.stat("input.mp4").st_size)
        else:
            args['init_video'] = None

        outputs = inference.run(**args)
        return [Path(out) for out in outputs]
