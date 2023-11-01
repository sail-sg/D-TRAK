# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import pickle

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

####
import torch
import random
import numpy as np

def set_seeds(seed):
    set_seed(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seeds(42)
####

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument(
        "--gen_path",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="TBD",
    )
    
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="TBD",
    )
    parser.add_argument(
        "--Z",
        type=int,
        default=None,
        help="TBD",
    )
    parser.add_argument(
        "--f",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument(
        "--t_strategy",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument("--e_seed", type=int, default=0, help="A seed for reproducible training.")
    
 
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "artbench": ("image", "label"),
}

# LABELS = ['art_nouveau', 'baroque', 'expressionism', 'impressionism', 'post_impressionism', 
#           'realism', 'renaissance', 'romanticism', 'surrealism', 'ukiyo_e']
# LABELS = [i.replace("_", " ").lower() for i in LABELS]

import pickle

def main():
    args = parse_args()
    ####
    gen_latents = np.memmap('{}/sd_gen_latents.npy'.format(args.gen_path), 
                                dtype=np.float32, 
                                mode='r', 
                                shape=(1000, 50,
                                       4, 32, 32))
    print(gen_latents.shape)
    ####
    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        set_seeds(42)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    local_files_only=True,
                                                    )
    ddim_noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
                                                            local_files_only=True,)
    ####
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, local_files_only=True,)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True,
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)
    ####
#     def compute_snr(timesteps):
#         """
#         Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
#         """
#         alphas_cumprod = noise_scheduler.alphas_cumprod
#         sqrt_alphas_cumprod = alphas_cumprod**0.5
#         sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

#         # Expand the tensors.
#         # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
#         sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
#         while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
#             sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
#         alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

#         sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
#         while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
#             sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
#         sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

#         # Compute SNR.z
#         snr = (alpha / sigma) ** 2
#         return snr
    ####
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to('cuda', dtype=weight_dtype)
    vae.to('cuda', dtype=weight_dtype)
    text_encoder.to('cuda', dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, 
                                                  cross_attention_dim=cross_attention_dim,
                                                  rank=128,
                                                 )

    unet.set_attn_processor(lora_attn_procs)
    ####
    # for n, p in unet.named_parameters():
    #     if p.requires_grad==True:
    #         print(n)
    ####
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # unet.enable_xformers_memory_efficient_attention()
            pass
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # data_files={'train': [args.dataset_name], }
    # dataset = load_dataset('imagefolder', 
    #                data_files=data_files, 
    # )
    ####
    import pandas as pd
    from datasets import DatasetDict, Dataset, load_dataset, Image

    if ("idx-train.pkl" in args.index_path) or ("idx-val.pkl" in args.index_path):
        df = pd.read_csv('../../codes/artbench/ArtBench-10.csv')
        df['path'] = df.apply(lambda x: "../../codes/artbench/data/artbench-10-imagefolder/{}/{}".format(
            x['label'], 
            x['name']), axis=1)
        # print(df.head())
        ####
        with open(args.index_path, 'rb') as handle:
            sub_idx = pickle.load(handle)
        print(sub_idx[0:5])
        sub_idx = sub_idx[args.split*1000:(args.split+1)*1000]
        print(sub_idx[0:5])
        ####
        dataset = DatasetDict({
            "train": Dataset.from_dict({
                "image": df.loc[sub_idx]['path'].tolist(),
                "label": df.loc[sub_idx]['label'].tolist(),
            }).cast_column("image", Image())
            ,})
        ####
    else:
        # pass
        df = pd.DataFrame()
        df['name'] = ['{}.png'.format(i) for i in range(1000)]
        df['label'] = ['ukiyo_e']*500+['post_impressionism']*500
        # df.head()
        df['path'] = df.apply(lambda x: "{}/{}".format(
            args.gen_path,
            # x['label'], 
            x['name']), axis=1)

        dataset = DatasetDict({
            "train": Dataset.from_dict({
                "image": df['path'].tolist(),
                "label": df['label'].tolist(),
            "idx": list(range(1000)),
            }).cast_column("image", Image())
            ,})
    ####
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            ####
            caption = 'a {} painting'.format(caption.replace("_", " ").lower())
            ####
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # Set the training transforms
    ####
    train_dataset = dataset["train"]
    ####
    train_dataset = train_dataset.with_transform(preprocess_train)
    print(len(train_dataset))
    print(train_dataset[0])
    ####

#     def collate_fn(examples):
#         pixel_values = torch.stack([example["pixel_values"] for example in examples])
#         pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#         input_ids = torch.stack([example["input_ids"] for example in examples])
#         return {"pixel_values": pixel_values, "input_ids": input_ids}

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids_list = []
        for example in examples:
            input_ids_list.append(example["input_ids"])

        input_ids = torch.stack(input_ids_list)

        idx = [example["idx"] for example in examples]

        return {"pixel_values": pixel_values, "input_ids": input_ids, "idx": idx}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    def count_parameters(model):
        return sum(p.numel() for p in unet.parameters() if p.requires_grad)
        
    # Scheduler and math around the number of training steps.
    model_path = '{}'.format(args.output_dir)
    print(model_path)
    if 'checkpoint-0' in args.output_dir:
        print(args.output_dir)
        unet.half()
        unet.cuda()
    else:
        print(args.output_dir)
        unet.load_attn_procs(model_path, weight_name="pytorch_model.bin")
        
    unet.eval()

    print(unet.dtype)
    print(count_parameters(unet))
    print(f"Number of parameters: {count_parameters(unet)//1e6:.2f}M")
    ####
    ####
    from trak.projectors import ProjectionType, AbstractProjector, CudaProjector
    projector = CudaProjector(grad_dim=count_parameters(unet), 
                          proj_dim=args.Z,
                          seed=42, 
                          proj_type=ProjectionType.normal,
                          # proj_type=ProjectionType.rademacher,
                          device='cuda:0')
    ####
    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad==True}
    ####
    import torch.nn.functional as F
    delattr(F, "scaled_dot_product_attention") # 很重要！！！！  
    print(hasattr(F, "scaled_dot_product_attention"))
    
    from torch.func import functional_call, vmap, grad 
    def vectorize_and_ignore_buffers(g, params_dict=None):
        """
        gradients are given as a tuple :code:`(grad_w0, grad_w1, ... grad_wp)` where
        :code:`p` is the number of weight matrices. each :code:`grad_wi` has shape
        :code:`[batch_size, ...]` this function flattens :code:`g` to have shape
        :code:`[batch_size, num_params]`.
        """
        batch_size = len(g[0])
        out = []
        if params_dict is not None:
            for b in range(batch_size):
                out.append(torch.cat([x[b].flatten() for i, x in enumerate(g) if is_not_buffer(i, params_dict)]))
        else:
            for b in range(batch_size):
                out.append(torch.cat([x[b].flatten() for x in g]))
        return torch.stack(out)
    ####
    if args.f=='mean-squared-l2-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            # predictions = predictions.reshape(1, -1)
            # f = torch.norm(predictions.float(), p=2.0, dim=-1)**2 # squared
            # f = f/predictions.size(1) # mean
            # f = f.mean()
            ####
            f = F.mse_loss(predictions.float(), torch.zeros_like(targets).float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='mean':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            f = predictions.float()
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='l1-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=1.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='l2-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=2.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='linf-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=float('inf'), dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    else:
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            f = F.mse_loss(predictions.float(), targets.float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            return f
    ####
    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, 
                              in_dims=(None, None, 0, 0, 0, 0,
                                       ),
                             )
    ####
    if "idx-train.pkl" in args.index_path:
        filename = os.path.join('{}/features-{}/sd-lora-train-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))
    elif "idx-val.pkl" in args.index_path:
        filename = os.path.join('{}/features-{}/sd-lora-val-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))
    else:
        # pass
        filename = os.path.join('{}/features-{}/sd-lora-gen-sampling-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    dstore_keys = np.memmap(filename, 
                            dtype=np.float32, 
                            mode='w+', 
                            shape=(len(train_dataset), args.Z))    
    
    for step, batch in enumerate(train_dataloader):
        set_seeds(42)
        print(step)
        
        for key in batch.keys():
            if key!='idx':
                batch[key] = batch[key].cuda()
        
        # Skip steps until we reach the resumed step
        
        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.mode()
            latents = latents * vae.config.scaling_factor
        
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            
        bsz = latents.shape[0]
        ####
        # if args.t_strategy=='uniform':
        #     selected_timesteps = range(0, 1000, 1000//args.K)
        # elif args.t_strategy=='cumulative':
        #     selected_timesteps = range(0, args.K)   
        ddim_noise_scheduler.set_timesteps(50, device='cuda')
        selected_timesteps = ddim_noise_scheduler.timesteps
        ####
        for index_t, t in enumerate(selected_timesteps):
            # Sample a random timestep for each image
            timesteps = torch.tensor([t]*bsz, device=latents.device)
            timesteps = timesteps.long()
            ####
            pred_original_sample = torch.from_numpy(gen_latents[batch['idx'], index_t]).half().cuda() # \hat{x_0}
            ####
            set_seeds(args.e_seed*1000+t) # !!!!
                    
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(pred_original_sample, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            ####
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, noisy_latents, timesteps, encoder_hidden_states,
                                                                 target,
                                                                )
            ft_per_sample_grads = vectorize_and_ignore_buffers(list(ft_per_sample_grads.values())) # 这个没啥问题啊    

            # print(ft_per_sample_grads.size())
            # print(ft_per_sample_grads.dtype)
            
            if index_t==0:
                emb = ft_per_sample_grads
            else:
                emb += ft_per_sample_grads
            # break
        emb = emb / args.K
        print(emb.size())
        emb = projector.project(emb, model_id=0) # ddpm
        print(emb.size())
        print(emb.dtype)
            
        # dstore_keys[step*args.train_batch_size:step*args.train_batch_size+bsz] = emb.detach().cpu().numpy()
        ####
        while (np.abs(dstore_keys[step*args.train_batch_size:step*args.train_batch_size+bsz, 0:32]).sum()==0):
            print('saving')
            dstore_keys[step*args.train_batch_size:step*args.train_batch_size+bsz] = emb.detach().cpu().numpy()
        ####
        
        print(step, t)
        print(step*args.train_batch_size, step*args.train_batch_size+bsz)
        # break

if __name__ == "__main__":
    main()


