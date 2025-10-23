import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from ConditionVideo.models.unet import UNet3DConditionModel
from ConditionVideo.data.dataset import ConditionVideoDataset
from ConditionVideo.pipelines.pipeline_conditionvideo import ConditionVideoPipeline
from ConditionVideo.util import save_videos_grid, ddim_inversion
from einops import rearrange
from datetime import datetime
import imageio
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def eval(device,seed,validation_pipeline,validation_data,global_step,output_dir,
        ddim_inv_scheduler,inv_latent,weight_dtype, sample_dataset,
        latent_idx_arr=None):
    samples = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    ddim_inv_latent = None
    if validation_data.use_inv_latent:
        ddim_inv_latent = ddim_inversion(
                validation_pipeline, ddim_inv_scheduler, video_latent=inv_latent,
                num_inv_steps=validation_data.num_inv_steps, prompt=validation_data.inv_prompt)[-1].to(weight_dtype)

    whole_conds = sample_dataset.load_conds(validation_data.whole_conds, \
        validation_data.whole_conds_video_path).to(weight_dtype) # (b f c h w)
    print(f"whole conds shape: {whole_conds.shape}")
    for idx, prompt in enumerate(validation_data.prompts):
        if validation_data.random: # random condition index in the video
            cond_idx = torch.randint(0, len(whole_conds), (1,)).item()
        elif validation_data.constant is not None: # constant condition index in the video
            cond_idx = validation_data.constant
        else:
            cond_idx = idx
        cond = whole_conds[cond_idx].unsqueeze(dim=0) # (b f c h w)
        os.makedirs(f"{output_dir}/conds/", exist_ok=True)
        save_videos_grid(rearrange(cond.to(weight_dtype),"b f c h w-> b c f h w"), f"{output_dir}/conds/{cond_idx}.gif")
        sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent, 
                                        cond=cond,**validation_data).videos
        assert sample is not None
        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}-{cond_idx}-{prompt}.gif")
        if len(samples) > 10:
            print("samples shape too large, only save 10 samples")
        else:
            samples.append(sample)

    samples = torch.concat(samples)
    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
    save_videos_grid(samples, save_path)
    logger.info(f"Saved no tuned samples to {save_path}")

def main(
    pretrained_model_path: str,
    output_dir: str,
    video_config: Dict,
    validation_data: Dict,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    cond_weight: float = 0.0,
    controlnet_path: str = None,
    prefix: str = "",
    cond_type: str = None,
    latent_idx_arr = None,
):
    # if key not in validation_data: # TODO:
    if "random" not in validation_data.keys() or validation_data["random"] is None:
        validation_data["random"] = False
        print("random not in validation_data, set to False")
    if "controlnet_inv_noise" not in validation_data.keys() or validation_data["controlnet_inv_noise"] is None:
        validation_data["controlnet_inv_noise"] = False
        print("controlnet_inv_noise not in validation_data, set to False") # whether to disentangle
    if "conrtolnet_inv_noise_idx" not in validation_data.keys() or validation_data["conrtolnet_inv_noise_idx"] is None:
        validation_data["conrtolnet_inv_noise_idx"] = 0 # when disentangle ends
        print("conrtolnet_inv_noise_start not in validation_data, set to 0")
    if "inv_prompt" not in validation_data.keys() or validation_data["inv_prompt"] is None:
        validation_data["inv_prompt"] = "" # prompt for customized video inversion
        print("inv_prompt not in validation_data, set to empty")

    time = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = os.path.join(output_dir, time+prefix)
        
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    from conditionvideo.models.controlnet import ControlNet3DModel
    controlnet = ControlNet3DModel.from_pretrained_2d(controlnet_path)

    controlnet.requires_grad_(False)
    pipeline_kwargs = {"controlnet": controlnet}
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Get the training dataset
    sample_dataset = ConditionVideoDataset(**video_config, cond_type=cond_type)

    # Get the validation pipeline
    validation_pipeline = ConditionVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        **pipeline_kwargs
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Prepare everything with our `accelerator`.
    unet, controlnet= accelerator.prepare(
                    unet, controlnet
                )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.cond_weight = cond_weight

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("conditionvideo-fine-tune")

    # Train!
    global_step = 0

    if validation_data.use_inv_latent: # inv_latent for customized scene
        print("get and save inv video")
        video_name = os.path.basename(validation_data.inv_video_path).split(".")[0]
        video_save_path = os.path.join(output_dir, f"inv_video/{video_name}.gif")
        inv_video = sample_dataset.get_video_and_save(validation_data.inv_video_path, 
                                        video_save_path).to(weight_dtype).unsqueeze(0)
        video_length = inv_video.shape[1]
        # get latent
        inv_video = inv_video.to(accelerator.device)
        inv_video = rearrange(inv_video, "b f c h w -> (b f) c h w")
        latents = vae.encode(inv_video).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        inv_latent = latents * 0.18215 # TODO: only one inv video
    else:
        inv_latent = None

    eval(device=accelerator.device, seed=seed,validation_pipeline=validation_pipeline, 
            validation_data = validation_data, global_step=global_step, output_dir=output_dir,ddim_inv_scheduler=ddim_inv_scheduler,inv_latent=inv_latent,
            weight_dtype=weight_dtype, sample_dataset=sample_dataset,latent_idx_arr=latent_idx_arr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
