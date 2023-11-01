import argparse
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import pickle

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
check_min_version("0.16.0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
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
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
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
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default="fp16",
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    #         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    #         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    #     ),
    # )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
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
    
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
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

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    print(args)
    ####
    gen_latents = np.memmap('{}/gen_latents.npy'.format(args.gen_path), 
                                dtype=np.float32, 
                                mode='r', 
                                shape=(1000, 50,
                                       3, 32, 32))
    print(gen_latents.shape)
    ####
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seeds(args.seed)
    ####
    print(args.model_config_name_or_path)
    config = UNet2DModel.load_config(args.model_config_name_or_path)
    config['resnet_time_scale_shift'] = 'scale_shift'
        
    model = UNet2DModel.from_config(config)
    print(model.dtype)
    ####
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    ddim_noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    ####
    if "idx-train.pkl" in args.index_path:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
        
        with open(args.index_path, 'rb') as handle:
            sub_idx = pickle.load(handle)
        print(sub_idx[0:5])
        sub_idx = sub_idx[args.split*1000:(args.split+1)*1000]
        print(sub_idx[0:5])
        dataset = dataset.select(sub_idx)
        
    elif "idx-val.pkl" in args.index_path:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="test",
        )
        
        with open(args.index_path, 'rb') as handle:
            sub_idx = pickle.load(handle)
        print(sub_idx[0:5])
        sub_idx = sub_idx[args.split*1000:(args.split+1)*1000]
        print(sub_idx[0:5])
        dataset = dataset.select(sub_idx)
        
    else:
        import pandas as pd
        df = pd.DataFrame()
        df['path'] = ['{}/{}.png'.format(args.gen_path, i) for i in range(1000)]
        
        from datasets import DatasetDict, Dataset, Image
        dataset = DatasetDict({
        "train": Dataset.from_dict({
            "img": df['path'].tolist(),
            "idx": list(range(1000)),
        }).cast_column("img", Image()),})
        dataset = dataset["train"]
    ####
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        # images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        
        idx = [idx for idx in examples["idx"]]

        return {"input": images,
                "idx": idx,
               }

    ####

    ####
    dataset.set_transform(transform_images)
    ####
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    # Scheduler and math around the number of training steps.
    model_path = '{}/unet/diffusion_pytorch_model.bin'.format(args.output_dir)
    print(model_path)
    if 'checkpoint-0' in args.output_dir:
        print(args.output_dir)
    else:
        print(args.output_dir)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    print(model.dtype)
    print(count_parameters(model))
    print(f"Number of parameters: {count_parameters(model)//1e6:.2f}M")
    ####
    ####
    from trak.projectors import ProjectionType, AbstractProjector, CudaProjector
    projector = CudaProjector(grad_dim=count_parameters(model), 
                          proj_dim=args.Z,
                          seed=42, 
                          proj_type=ProjectionType.normal,
                          # proj_type=ProjectionType.rademacher,
                          device='cuda:0')
    ####
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad==True}
    ####
    import torch.nn.functional as F
    delattr(F, "scaled_dot_product_attention") #  
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, })
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
                              in_dims=(None, None, 0, 0, 0, 
                                       ),
                             )
    ####
    if "idx-train.pkl" in args.index_path:
        filename = os.path.join('{}/features-{}/ddpm-train-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))
    elif "idx-val.pkl" in args.index_path:
        filename = os.path.join('{}/features-{}/ddpm-val-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))
    else:
        filename = os.path.join('{}/features-{}/ddpm-gen-sampling-troj-keys-{}-{}-{}-{}-{}.npy'.format(
            args.output_dir, 
            args.e_seed, 
            args.split, args.K, args.Z, args.f, args.t_strategy))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    dstore_keys = np.memmap(filename, 
                            dtype=np.float32, 
                            mode='w+', 
                            shape=(len(dataset), args.Z))    
    
    for step, batch in enumerate(train_dataloader):
        set_seeds(42)
        for key in batch.keys():
            if key!='idx':
                batch[key] = batch[key].cuda()
            
        # Skip steps until we reach the resumed step
        latents = batch["input"]         
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
            pred_original_sample = torch.from_numpy(gen_latents[batch['idx'], index_t]).cuda() # \hat{x_0}
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
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, noisy_latents, timesteps, 
                                                                 target,
                                                                )
            ft_per_sample_grads = vectorize_and_ignore_buffers(list(ft_per_sample_grads.values())) #     

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