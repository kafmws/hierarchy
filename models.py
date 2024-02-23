import os
import sys
import torch
import torchvision.transforms as T

from PIL import Image
from typing import List, Union
from transformers import AutoModel, AutoConfig
from torchvision.transforms import InterpolationMode
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer

from eva_clip import create_model_and_transforms, get_tokenizer
from clip import clip


def openai_clip(arch: str):
    model, preprocess = clip.load(arch, 'cpu')
    tokenizer = clip.tokenize
    return model, tokenizer, preprocess


def eva_clip(arch: str):
    """EVA-CLIP"""
    # arch = "EVA02-CLIP-B-16"  # in eva_clip.pretrained._PRETRAINED
    pretrained = "eva_clip"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"  source of weights
    
    tokenizer = get_tokenizer(arch)  # auto truncate
    model, _, preprocess = create_model_and_transforms(arch, pretrained, force_custom_clip=True)
    
    # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
    
    return model, tokenizer, preprocess


def hf_clips(arch_or_path: str):

    # arch_or_path = "BAAI/EVA-CLIP-8B"  # or /path/to/local/EVA-CLIP-8B

    # you can also directly use the image processor by torchvision
    # squash
    # processor = T.Compose(
    #     [
    #         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    #         T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    #         T.ToTensor(),
    #         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    #     ]
    # )
    # shortest
    # processor = T.Compose(
    #     [
    #         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    #         T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
    #         T.CenterCrop(image_size),
    #         T.ToTensor(),
    #         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    #     ]
    # )

    model = AutoModel.from_pretrained(
        arch_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True).eval()
    image_size = model.config.vision_config.image_size
    print(f'input image size for {arch_or_path}: {image_size}')
    
    _tokenizer = CLIPTokenizer.from_pretrained(arch_or_path)
    _preprocess = CLIPImageProcessor(size={"shortest_edge": image_size}, do_center_crop=True, crop_size=image_size)
    
    # input_ids = _tokenizer(["a diagram", "a dog", "a cat"], return_tensors="pt", padding=True).input_ids.to('cuda')
    # input_pixels = _preprocess(images=Image.open(image_path), return_tensors="pt", padding=True).pixel_values.to('cuda')
    
    def tokenizer(texts: List[str], context_length: int = 77):
        return _tokenizer(texts, return_tensors="pt", padding=True, context_length=context_length).input_ids
    
    def preprocess(images):
        return _preprocess(images=images, return_tensors="pt", padding=True).pixel_values
    
    return model, tokenizer, preprocess


class ConvNet(torch.nn.Module):
    
    pass


def get_model(model_name: str, arch: str):
    m = {
        'openai_clip': openai_clip,
        'eva_clip': eva_clip,
    }
    
    load = m[model_name] if model_name in m else hf_clips
    model, tokenizer, preprocess = load(arch)
    
    return model, tokenizer, preprocess
