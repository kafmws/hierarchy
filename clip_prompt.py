import os
import sys
import json
import shutil
import timeit

# for PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([project_root])

# for reproducibility
from utils import set_seed

seed = 42
set_seed(seed)

import cv2
import torch
import pickle
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import accumulate
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.utils import draw_bounding_boxes
from typing import Counter, List, Tuple, Any, Iterable
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import get_model
from hierarchical.hierarchy import get_hierarchy
from dataset import get_dataset, get_feature_dataset
from prompts import clsname2prompt, hierarchical_prompt


prompt_base = '/root/projects/readings/data'


def generate_dataset():
    # 生成原始iwildcam图像，画框iwildcam图像和背景模糊图像

    # 读入bbx, 建立图像名到bbx的映射
    img2dets = {}
    md_result = json.load(open('/root/projects/readings/data/wilds_json_01.json', 'r'))
    for imginfo in md_result['images']:
        img2dets[imginfo['file'].split('/')[-1]] = imginfo['detections']

    def draw_bbx(image, bbxinfo, savepath):
        # imgpath = '/data/wilds/iwildcam_v2.0/train/95f28cb2-21bc-11ea-a13a-137349068a90.jpg'
        # image = Image.open(imgpath)

        bbxes = []
        for bbx in bbxinfo:
            if bbx['category'] != '1' or bbx['conf'] < 0.1:  # not animal
                continue
            bbx = bbx['bbox']
            bbx[0] = bbx[0] * image.width
            bbx[1] = bbx[1] * image.height
            bbx[2] = bbx[2] * image.width + bbx[0]
            bbx[3] = bbx[3] * image.height + bbx[1]

            bbxes.append(bbx)

        assert len(bbxes)
        try:
            # 定义bounding box左上点和右下点的坐标
            bbox = torch.tensor(bbxes)
            # 将图像转换为Tensor
            image_tensor = transforms.ToTensor()(image).mul(255).byte()
            # 绘制bounding box
            result_image = draw_bounding_boxes(image_tensor, bbox, colors='red', width=2)
            # 将Tensor转换回图像
            result_image = transforms.ToPILImage()(result_image)
            # 显示图像和bounding box
            # result_image.show()
            result_image.save(savepath)
            # print(savepath)
        except:
            print(bbxes)
            print(imgpath)

        # {prompt_base}/bbx/loxodonta_africana/9294f582-21bc-11ea-a13a-137349068a90.jpg

    def get_fg_only_img(image, maskpath, savepath):
        mask = Image.open(maskpath).convert('L')
        mask = mask.resize(image.size, Image.NEAREST)
        # 将掩码转换为NumPy数组
        mask_np = np.array(mask)

        # 使用掩码提取物体
        object_image_np = np.array(image)
        object_image_np[mask_np == 0] = 0
        object_image = Image.fromarray(object_image_np)

        # 保存新的JPEG图像
        object_image.save(savepath)
        # print(savepath)

    def blur_background(imagepath, maskpath, savepath, sigma=5):
        image = cv2.imread(imagepath)  # 读取图像
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        # 使用掩码将图像中的物体和非物体分开
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        object_region = cv2.bitwise_and(image, image, mask=mask)
        non_object_region = cv2.bitwise_and(image, image, mask=~mask)

        # 创建高斯模糊核
        # kernel_size = int(6 * sigma + 1)
        # blur_kernel = cv2.getGaussianKernel(kernel_size, sigma)

        # 对非物体区域进行高斯模糊
        # blurred_non_object_region = cv2.filter2D(non_object_region, -1, blur_kernel)
        blurred_non_object_region = cv2.blur(non_object_region, (15, 15))
        blurred_non_object_region[mask.astype(bool)] = 0

        # 将处理后的非物体区域与物体区域合并
        result = cv2.add(object_region, blurred_non_object_region)
        # result = cv2.bitwise_or(object_region, blurred_non_object_region)
        cv2.imwrite(savepath, result)

    def gray_bg(imagepath, mask, savepath):
        image = cv2.imread(imagepath)  # 读取图像
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        # 使用掩码将图像中的物体和非物体分开
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        object_region = cv2.bitwise_and(image, image, mask=mask)
        non_object_region = cv2.bitwise_and(image, image, mask=~mask)

        # 对非物体区域去色
        non_object_region = cv2.cvtColor(non_object_region, cv2.COLOR_BGR2GRAY)
        non_object_region = cv2.cvtColor(non_object_region, cv2.COLOR_GRAY2BGR)  # 转为相同通道
        non_object_region[mask.astype(bool)] = 0

        # 将处理后的非物体区域与物体区域合并
        result = cv2.add(object_region, non_object_region)
        cv2.imwrite(savepath, result)

    def draw_bbx_cv2(img, savepath):
        img = '/data/wilds/iwildcam_v2.0/train/95f28cb2-21bc-11ea-a13a-137349068a90.jpg'
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        for bbx in img2dets[fname]:
            if bbx['category'] != '1' or bbx['conf'] < 0.05:  # not animal
                continue
            bbox = bbx['bbox']

            # 计算bounding box在图像上的坐标
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            bbxes.append(bbx)

            fig, ax = plt.subplots()

            # 绘制bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.save(savepath)

    iwildcambase = '/data/kafm/wilds36/test'
    for species in os.listdir('/data/wilds_sq336_36/test/'):
        os.makedirs(f'{prompt_base}/original/{species}', exist_ok=True)
        os.makedirs(f'{prompt_base}/bbx/{species}', exist_ok=True)
        os.makedirs(f'{prompt_base}/gaussian_bg/{species}', exist_ok=True)
        os.makedirs(f'{prompt_base}/fg_only/{species}', exist_ok=True)
        os.makedirs(f'{prompt_base}/gray_bg/{species}', exist_ok=True)

        # not_found = '/data/wilds_sq336_36/test/pecari_tajacu/p_11620/s_56474/2e7bc21e-d3c3-11eb-bd40-653a79b12701.jpg'
        for base, dirs, files in os.walk(f'/data/wilds_sq336_36/test/{species}/'):
            for fname in files:
                imgpath = f'{base.replace("/data/wilds_sq336_36/test", iwildcambase)}/{fname}'
                new_path = f'{prompt_base}/original/{species}/{fname}'
                maskpath = f'/data/wilds/iwildcam_v2.0/2022_instance_masks/{fname.replace(".jpg", ".png")}'

                shutil.copy(imgpath, new_path)
                continue

                image = Image.open(new_path)
                if os.path.exists(maskpath):
                    # draw_bbx(image, img2dets[fname], f'{prompt_base}/bbx/{species}/{fname}')
                    # get_fg_only_img(image, maskpath, f'{prompt_base}/fg_only/{species}/{fname}')
                    blur_background(new_path, maskpath, f'{prompt_base}/gaussian_bg/{species}/{fname}', sigma=10)
                    # gray_bg(new_path, maskpath, f'{prompt_base}/gray_bg/{species}/{fname}')
                else:
                    # shutil.copy(new_path, f'{prompt_base}/bbx/{species}/{fname}')
                    # shutil.copy(new_path, f'{prompt_base}/fg_only/{species}/{fname}')
                    shutil.copy(new_path, f'{prompt_base}/gaussian_bg/{species}/{fname}')
                    # shutil.copy(new_path, f'{prompt_base}/gray_bg/{species}/{fname}')


# generate_dataset()


class VisualPromptDataSet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        prompt: str,
        transform=None,
        target_transform=None,
    ):
        super().__init__(f'{root}/{prompt}', transform, target_transform)
        self.prompt = prompt
        self.n_classes = len(self.classes)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        sample, target = super().__getitem__(index=index)
        path, _ = self.samples[index]
        return sample, target, path


# config
# OPENAI-CLIP
model_name, arch = 'openai_clip', 'ViT-L/14@336px'

# EVA-CLIP  大小写敏感
# model_name, arch = 'eva_clip', 'EVA01-CLIP-g-14'
# model_name, arch = 'eva_clip', 'EVA02-CLIP-L-14-336'
# model_name, arch = 'eva_clip', 'EVA02-CLIP-bigE-14-plus'

# EVA-CLIP-8B
# model_name, arch = 'eva_clip_8B', 'BAAI/EVA-CLIP-8B'
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'

# datasets = {'imagenet1k': ['val']}
datasets = {'iwildcam36': ['test'], 'aircraft': ['test'], 'animal90': ['test']}

output_dir = project_root
feat_output_dir = os.path.join(output_dir, 'feature_dataset', model_name, arch.replace("/", "-"))
os.makedirs(feat_output_dir, exist_ok=True)

pic_output_dir = os.path.join(output_dir, 'pic')
os.makedirs(pic_output_dir, exist_ok=True)


supported_visual_prompt = ['original', 'fg_only', 'gaussian_bg', 'gray_bg', 'crop', 'bbx', 'train', 'crop_train']
supported_text_prompt = [
    'vanilla',
    'english_name',
    'context',
    'category context',
    'image context',
    'image category context',
    'night',
    'description',
]


def clip_zeroshot(model, tokenizer, visual_prompt, text_prompt, transform, target_transform=None, logits_path=''):
    assert visual_prompt in supported_visual_prompt
    assert text_prompt in supported_text_prompt

    # 填充text模板
    h = get_hierarchy('iwildcam36')
    classes = h.get_leaves()
    classes.sort(key=lambda node: node.inlayer_idx)

    texts = []
    if text_prompt == 'description':
        texts = [
            'a bird with a dark blue body, white spots, black crest, red throat wattle, long tail with black stripes.',
            'an antelope with reddish brown fur, white belly and tail, curved horns.',
            'a monkey with black or dark brown fur and a white beard on its throat and ears. It has a red or orange',
            'a large pheasant with a blue head and neck, a brown body, and a long tail.',
            'a porcupine with black or dark brown fur and a yellow brush tail, red or orange patches.',
            'a large and cloven-hooved herbivore with short hair, hollow horns, long tufted',
            'an even-toed ungulate in the camel family with a brown or gray body, a long curved neck, a narrow chest and a hump.',
            'a bovid animal with reddish brown or chestnut fur and a black stripe on its head.',
            'a bird with black plumage, a forward-curled crest, a white belly and undertail, and a yellow knob on its bill.',
            'the Gambian pouched rat, a nocturnal rodent with coarse brown fur, dark eye rings and large cheek.',
            'a large rodent with dark brown fur and white spots, short ears and barely visible tail.',
            'a rodent with coarse, glossy fur that ranges from orange to black, and a contrasting rump.',
            'an odd-toed ungulate in the horse family with a white body, black stripes, a large head, round ears.',
            'plains zebra, with black and white stripes.',
            'a rodent in the squirrel family with a brown body, black stripes, a large head, large eyes.',
            'the giraffe, with extremely long neck and legs, short horns on the head, spotted patterns on the body.',
            'a medium-sized, spotted wild cat with a grayish-brown head and neck, and a reddish-brown to yellowish body.',
            'an African bush elephant, the largest land animal, with grey skin, large ears and long tusks.',
            'a dik-dik. It has yellowish-gray or reddish-brown fur, black hooves, small and long head.',
            'a small deer with gray-brown or reddish-brown fur, white underparts, and short antlers in males.',
            'central American red brocket.',
            'a bird with bronze or green plumage, an eye-spotted tail, a bare blue or red head and neck.',
            'a deer species. It has a short brownish or grayish coat, sometimes with creamy markings.',
            'a carnivore with a brown or yellow body, coarse fur, a white ringed nose and white eye patches on its head.',
            'a medium-sized deer with reddish-brown or grayish-brown fur, white underparts.',
            'a large cat species with a yellowish to tan fur with black spots and rosettes.',
            'a baboon, with a dark olive-gray fur that is longer in males, forming a mane, a flat head.',
            'a pig-like animal with coarse, dark brown fur and a white collar around its neck.',
            'a bird in the guan family with a dark brown body, white spots, a black crest on its head.',
            'a pheasant-like bird with dark red or brown feathers and a grey head. It has a red bill and legs.',
            'a large brownish cat with a long tail and retractable claws.',
            'an African buffalo, with black or brown fur, thick bone shield on the head and shoulders, curved horns.',
            'a hoofed animal with dark brown or black fur, white fringes on its ears, lips, throat and chest.',
            'a pig-like animal with dark hair, white lips and throat, and a musky odor.',
            'a brownish to grayish plumage, a short tail, and a long bill.',
            'a carnivore with gray or black fur, a striped head and neck, a black-tipped tail, and reddish-brown legs and sides.',
        ]
    for i, node in enumerate(classes):
        if text_prompt == 'vanilla':
            # a photo of a {class}.
            texts.append(f'a photo of a {node.name}.')

        if text_prompt == 'english_name':
            # a photo of a {english name}.
            texts.append(f'a photo of a {node.english_name}.')

        if text_prompt == 'description':
            # a photo of a {english name} + description
            texts[i] = f'a photo of a {node.english_name}, ' + texts[i]

        if text_prompt == 'night':
            # a photo of a {english name}.
            texts.append(f'a photo of a {node.english_name} in the night.')

        if text_prompt == 'context':
            # a photo of a {class} in the wild.
            texts.append(f'a photo of a {node.english_name} in the wild.')

        if text_prompt == 'category context':
            # a photo of a {class}, a kind of animal.
            texts.append(f'a photo of a {node.english_name}, a kind of animal.')

        if text_prompt == 'image context':
            # a camera trap image of a {class}.
            texts.append(f'a camera trap image of a {node.english_name}.')

        if text_prompt == 'image category context':
            # a camera trap image of a {class}.
            texts.append(f'a camera trap image of a {node.english_name}, a kind of animal.')

    text_features = encode_text_batch(model, tokenizer, texts=texts, n_classes=len(texts), text_fusion=False, device=device)

    dataset = VisualPromptDataSet(root=prompt_base, prompt=visual_prompt, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=64)

    saved_logits = np.array([[0] * len(texts)])
    labels = np.array([0])
    correct = 0
    with torch.no_grad():
        for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
            targets: torch.Tensor = targets.to(device)
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            similarity = (100.0 * logits).softmax(dim=-1)

            predicts = similarity.topk(1, dim=1).indices.squeeze(dim=1)
            correct += sum(torch.eq(targets, predicts)).item()

            saved_logits = np.concatenate([saved_logits, logits.cpu().numpy()], axis=0)
            labels = np.concatenate([labels, targets.cpu().numpy()], axis=0)

            for pred, label, path in zip(predicts, targets, paths):
                if pred != label:
                    new_name = (
                        f'{classes[label.item()].english_name}_({classes[pred.item()].english_name})_{path.split("/")[-1]}'
                    )
                    shutil.copy(path, f'/root/projects/readings/wildlife_interesting/clip_error/{new_name}')
                    # if '88d92c16-21bc-11ea-a13a-137349068a90.jpg' in path:
                    # print(f'from {classes[label.item()].english_name} to {classes[pred.item()].english_name}')
                    # if '9312b940-21bc-11ea-a13a-137349068a90.jpg' in path:
                    # print(f'from {classes[label.item()].english_name} to {classes[pred.item()].english_name}')

    print(f'{visual_prompt} {text_prompt}: {correct / len(loader.dataset) * 100: .2f}')

    if logits_path:
        np.save(f'{model_name}_{logits_path}_logits.npy', saved_logits[1:])
        np.save(f'{model_name}_{logits_path}_labels.npy', labels[1:])


def collect_image_features(model):
    for dataset_name, splits in datasets.items():
        for split in splits:
            output_filepath = os.path.join(feat_output_dir, f'{dataset_name}_{split}_features.pkl')
            if os.path.isfile(output_filepath):
                print(f'{output_filepath} exists, skip!')
                continue

            print(f'preparing {dataset_name} {split} features...')
            dataset = get_dataset(dataset_name=dataset_name, split=split, transform=preprocess)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=2)

            sample_features = []  # [path, target, feature]
            path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

            # calculate features
            with torch.no_grad():
                for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                    images = images.to(device)
                    # with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(images)
                    for (
                        path,
                        target,
                        feature,
                    ) in zip(paths, targets, image_features):
                        sample_features.append((path[path_prefix_len:], target.item(), feature.cpu().numpy()))

            data = {
                'fname': f'{dataset_name}_{split}_features.pkl',
                'model': f'{model_name}:{arch}',
                'features': sample_features,
                'transform': str(preprocess),
            }

            # torch.save(obj=data, f=os.path.join(feat_output_dir, data['fname'] + '.pth'))
            with open(output_filepath, 'wb') as file:
                pickle.dump(obj=data, file=file)


def encode_text_batch(model, tokenizer, texts: List[str], n_classes, text_fusion, device):
    # print(texts)  # TODO: 查看tokenizer中大小写到底有没有变化
    multiple = len(texts) / n_classes
    assert multiple == int(multiple)
    multiple = int(multiple)

    text_features = []
    # calculate text features
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)  # tokenize
        _batch = list(range(0, len(tokens), multiple))
        if _batch[-1] < len(tokens):
            _batch.append(len(tokens))
        for endi in range(1, len(_batch)):
            batch_text_features = model.encode_text(tokens[_batch[endi - 1] : _batch[endi]])
            batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
            if text_fusion:
                batch_text_features = batch_text_features.mean(dim=0)
                batch_text_features /= batch_text_features.norm()
                batch_text_features = [batch_text_features]
            text_features.extend(batch_text_features)
    text_features = torch.stack(text_features, dim=0)
    assert (
        text_features.shape[0] == n_classes if text_fusion else len(tokens)
    ), f'{text_features.shape[0]} != {n_classes if text_fusion else len(tokens)}'
    return text_features


def collect_clip_logits(tokenizer, text_fusion=False, only: int = None, dump=False):
    """CLIP zero-shot inference with multiple prompts.

    Args:
        text_fusion (bool, optional): To mean text prompt embedding or not. Defaults to False.
        only (int, optional): Test all kinds of prompts or which only. Defaults to None (test all).
        dump (bool, optional): Whether save the logits or not. Defaults to True.
    """

    dataset = get_feature_dataset(dataset_name='imagenet1k', split='val', model=model_name, arch=arch)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=2)
    print(f'preparing {dataset.dataset_name} {dataset.split} logits...')

    sample_logits = []  # [path, target, feature]
    path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

    # best prompt #3 w/o ensembling, #4 w/ ensembling
    n_classes = len(dataset.classes)
    text_inputs = clsname2prompt(dataset.dataset_name, dataset.classes)
    if only:
        text_inputs = text_inputs[only : only + 1]

    for i, texts in enumerate(text_inputs):
        text_features = encode_text_batch(
            model, tokenizer, texts=texts, n_classes=n_classes, text_fusion=text_fusion, device=device
        )
        # if not len(texts) > n_classes and not text_fusion:  # multiple == 1 also performs norm
        #     text_features /= text_features.norm(dim=-1, keepdim=True)

        correct = 0
        with torch.no_grad():
            for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                targets: torch.Tensor = targets.to(device)
                image_features: torch.Tensor = images.to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                similarity = (100.0 * logits).softmax(dim=-1)
                for (
                    path,
                    target,
                    _logits,
                ) in zip(paths, targets, logits):
                    sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))
                predicts = similarity.topk(1, dim=1).indices.squeeze(dim=1)
                if (
                    not text_fusion and len(texts) > n_classes
                ):  # multiple template for each class and not fuse the text embedding
                    predicts = predicts / (len(texts) / n_classes)

                """ correct two `sunglasses` classes, which trivially contributes to 0.06% accuracy"""
                # predicts = torch.where(predicts == 836, 1000, predicts)
                # predicts = torch.where(predicts == 837, 1000, predicts)
                # targets = torch.where(targets == 836, 1000, targets)
                # targets = torch.where(targets == 837, 1000, targets)

                correct += sum(torch.eq(targets, predicts)).item()

        print(f'accuracy of prompt #{i}: {correct / len(dataset) * 100: .2f}')

        if dump:
            data = {
                'fname': f'{dataset.dataset_name}_{dataset.split}_logits.pkl',
                'model': f'{model_name}:{arch}',
                'text_features': text_features,
                'logits': sample_logits,
                'texts': texts,
                'seed': seed,
            }

            with open(os.path.join(feat_output_dir, data['fname']), 'wb') as file:
                pickle.dump(obj=data, file=file)


def hierarchical_inference(tokenizer, dataset, selected_layers=None, n_thought_path=1, detailed=False):
    # soft_decision = True
    soft_decision = n_thought_path > 1

    feat_output_dir = f'{output_dir}/{model_name}/{arch.replace("/", "-")}'
    os.makedirs(feat_output_dir, exist_ok=True)

    sample_logits = []  # [path, target, feature]
    path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

    text_inputs, (isa_mask, layer_isa_mask, layer_mask), hierarchical_targets, classnames, h = hierarchical_prompt(
        dataset.dataset_name
    )
    layer_cnt, layer2name = h.layer_cnt, h.layer2name
    isa_mask = torch.from_numpy(isa_mask).to(device)
    layer_mask = torch.from_numpy(layer_mask).to(device)
    layer_isa_mask = torch.from_numpy(layer_isa_mask).to(device)
    flat_classnames = [item for sublist in classnames for item in sublist]
    layer_offset = [0] + list(accumulate(layer_cnt))

    row = arch[:-2] + (' HI' if selected_layers else ' ZS') + '     &' + '&'.join(['{:^9}'] * len(layer_cnt)) + '\\\\'

    for i, texts in enumerate(text_inputs):
        print(f'for prompts {i}#')

        text_features = encode_text_batch(
            model, tokenizer, texts=texts, n_classes=layer_offset[-1], text_fusion=False, device=device
        )

        # collect all results
        if not selected_layers:
            correct = [0] * len(layer_cnt)
            preds = [[] for _ in range(len(layer_cnt))]
            labels = [[] for _ in range(len(layer_cnt))]
        else:
            consistency = [0] * len(selected_layers)
            correct = [[0] * len(layer_cnt) for _ in range(len(selected_layers))]
            preds = [[[] for _ in range(len(layer_cnt))] for _ in range(len(selected_layers))]
            labels = [[[] for _ in range(len(layer_cnt))] for _ in range(len(selected_layers))]

        start_time = timeit.default_timer()
        with torch.no_grad():
            # the method also can be paralleled in batches
            # for images, targets, paths in tqdm(dataset, ncols=60, dynamic_ncols=True):
            for images, targets, paths in dataset:
                # layerify target
                layer_targets = hierarchical_targets[targets]
                targets = torch.LongTensor(layer_targets)

                targets: torch.Tensor = targets.to(device)
                image_features: torch.Tensor = images.to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                similarity = (100.0 * logits).softmax(dim=-1)
                similarity += 1e-6  # for logic consistency, considered all zero logits
                # for path, target, _logits, in zip(paths, targets, logits):
                #     sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))

                # thinking in hierarchy
                if selected_layers:
                    for i, selected_layer in enumerate(selected_layers):
                        consistent = True  # for soft decision only
                        topk, predicts, last_pred = None, None, None  # satisfies the code analyzer
                        for layer in selected_layer:
                            # if i == 13:
                            #     print('debug')

                            if layer == selected_layer[0]:  # the first thinking at 1st layer
                                mask = layer_mask[layer]
                            else:  # thoughts based on last thought
                                if soft_decision:  # soft decision have multiple path of thoughts
                                    # TODO: score scaling/processing
                                    # 归一化topk.values 或者该层的logits，温度可以由均值决定。
                                    # scores = torch.softmax(topk.values / topk.values.mean(), dim=-1)
                                    scores = topk.values / topk.values.mean()
                                    scores = torch.softmax(scores / 0.5, dim=-1)  # 小值尖锐，大值平滑
                                    # TODO: 先进行缩放，再调整温度
                                    mask = 0
                                    for pred, score in zip(topk.indices, scores):
                                        mask += layer_isa_mask[pred][layer] * score
                                else:  # hard decision
                                    mask = layer_isa_mask[predicts.item()][layer]
                            last_pred = predicts
                            # topk = (similarity * mask).topk(n_thought_path, dim=-1)
                            # topk = (similarity * mask).topk((int(sum(mask > 0) / 2) + 1) if soft_decision else 1, dim=-1)
                            topk = (similarity * mask).topk(int(sum(mask > 0)) if soft_decision else 1, dim=-1)
                            predicts = topk.indices
                            if last_pred is not None:
                                consistent = isa_mask[last_pred[0]][predicts[0]]
                                # if not consistent:
                                # print(f'{paths}: {flat_classnames[last_pred[0]]}/{last_pred[0]}=>{flat_classnames[predicts[0]]}/{predicts[0]}')
                            correct[i][layer] += sum(torch.eq(targets[layer], predicts[0])).item()

                            preds[i][layer].append(predicts[0] - layer_offset[layer])
                            labels[i][layer].append(targets[layer].item() - layer_offset[layer])

                        # compute hiearchical consistency for soft decision
                        consistency[i] += consistent

                else:
                    for layer in range(0, len(layer_cnt)):
                        masked_similarity = similarity * layer_mask[layer]
                        predicts = masked_similarity.topk(1, dim=-1).indices
                        correct[layer] += sum(torch.eq(targets[layer], predicts)).item()

                        preds[layer].append(predicts.item() - layer_offset[layer])
                        labels[layer].append(targets[layer].item() - layer_offset[layer])
        end_time = timeit.default_timer()
        print(f'{dataset.dataset_name} hiearchical inference time: {end_time - start_time :.2f}s')

        # latex accuracy output
        if selected_layers:
            for i, selected_layer in enumerate(selected_layers):
                res = [
                    '{:.2f}'.format(correct[i][idx] * 100 / len(dataset)) if idx in selected_layer else '-'
                    for idx in range(len(layer_cnt))
                ]
                print(row.format(*res) + (f' consistency: {consistency[i] * 100 / len(dataset): .2f}' if soft_decision else ''))
        else:
            res = ['{:.2f}'.format(correct[idx] * 100 / len(dataset)) for idx in range(len(layer_cnt))]
            print(row.format(*res))

        # detailed results
        if detailed:
            if selected_layers:
                for i, selected_layer in enumerate(selected_layers):
                    pass
            else:
                for layer in range(0, len(layer_cnt)):
                    report = classification_report(y_true=labels[layer], y_pred=preds[layer], target_names=classnames[layer])
                    # print(report)

                    # debug
                    diff = np.where(np.array(labels[layer]) != np.array(preds[layer]))
                    errs = np.array(dataset.targets)[diff]
                    errs = list(map(lambda idx: dataset.classes[idx], errs))
                    cnt = list(Counter(errs).items())
                    cnt.sort(key=lambda item: item[1], reverse=True)
                    print(cnt)

                    cm = confusion_matrix(y_true=labels[layer], y_pred=preds[layer], normalize='true')
                    disp = ConfusionMatrixDisplay(cm, display_labels=classnames[layer])
                    disp.plot(cmap='Blues', values_format='.2%', text_kw={'size': 6})
                    figpath = os.path.join(pic_output_dir, f'{dataset.dataset_name}_{layer2name[layer]}.png')
                    title = f'confusion matrix of {layer2name[layer]} layer'
                    plt.title(label=title)
                    plt.tight_layout()
                    plt.savefig(figpath, dpi=300)


if __name__ == '__main__':
    # Load the model
    model, tokenizer, preprocess = get_model(model_name, arch, device)
    # print(model.logit_scale.exp())
    # os._exit(0)

    clip_transform = Compose([torch.HalfTensor])
    clip_target_transform = Compose([torch.as_tensor])

    # clip_zeroshot(model, tokenizer, visual_prompt='crop_train', text_prompt='image category context',  # english_name
    #               transform=preprocess, logits_path='crop_train')
    clip_zeroshot(
        model,
        tokenizer,
        visual_prompt='crop',
        text_prompt='image category context',  # english_name
        transform=preprocess,
        logits_path='crop',
    )

    # for visual_prompt in supported_visual_prompt:
    # clip_zeroshot(model, tokenizer, visual_prompt=visual_prompt, text_prompt='description', transform=preprocess)

    # for visual_prompt in supported_visual_prompt:
    # for text_prompt in supported_text_prompt:
    # clip_zeroshot(model, tokenizer, visual_prompt=visual_prompt, text_prompt=text_prompt, transform=preprocess)

    # collect_image_features(model)
    # collect_clip_logits(tokenizer, text_fusion=True, dump=False)
    # collect_clip_logits(tokenizer, text_fusion=True, only=3, dump=False)

    def test_iwildcam36():
        dataset = get_feature_dataset(dataset_name='iwildcam36', split='test', model=model_name, arch=arch)
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(tokenizer, dataset)
        hierarchical_inference(
            tokenizer,
            dataset=dataset,
            selected_layers=[
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [0, 1, 2, 4],
                [0, 1, 4],
                [0, 2, 4],
                [0, 3, 4],
                [2, 3, 4],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
                [4],
            ],
        )

    def test_animal90():
        dataset = get_feature_dataset(
            dataset_name='animal90',
            split='test',
            model=model_name,
            arch=arch,
            transform=clip_transform,
            target_transform=clip_target_transform,
        )
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(tokenizer, dataset)

    def test_aircraft():
        dataset = get_feature_dataset(
            dataset_name='aircraft',
            split='test',
            model=model_name,
            arch=arch,
            transform=clip_transform,
            target_transform=clip_target_transform,
        )
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(tokenizer, dataset)
        hierarchical_inference(
            tokenizer,
            dataset=dataset,
            selected_layers=[
                [0, 1, 2],
                [0, 1],
                [0, 2],
                [1, 2],
                [2],
            ],
        )

    # test_iwildcam36()
    # test_aircraft()
    # test_animal90()
