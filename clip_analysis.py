import os
import sys
import timeit

# for PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([project_root])

# for reproducibility
from utils import set_seed

seed = 42
set_seed(seed)

import torch
import pickle
import numpy as np
from tqdm import tqdm
from itertools import accumulate
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sklearn.metrics import classification_report
from typing import Counter, List, Tuple, Any, Iterable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import get_model
from hierarchical.hierarchy import get_hierarchy
from dataset import get_dataset, get_feature_dataset
from prompts import clsname2prompt, hierarchical_prompt

# config
# OPENAI-CLIP
model_name, arch = 'openai_clip', 'ViT-L/14@336px'

# EVA-CLIP  大小写敏感
# model_name, arch = 'eva_clip', 'EVA01-CLIP-g-14'
# model_name, arch = 'eva_clip', 'EVA02-CLIP-L-14-336'
# model_name, arch = 'eva_clip', 'EVA02-CLIP-bigE-14-plus'

# EVA-CLIP-8B
# model_name, arch = 'eva_clip_8B', 'BAAI/EVA-CLIP-8B'
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# datasets = {'imagenet1k': ['val']}
datasets = {'iwildcam36': ['test'], 'aircraft': ['test'], 'animal90': ['test']}

output_dir = project_root
feat_output_dir = os.path.join(output_dir, 'feature_dataset', model_name, arch.replace("/", "-"))
os.makedirs(feat_output_dir, exist_ok=True)

pic_output_dir = os.path.join(output_dir, 'pic')
os.makedirs(pic_output_dir, exist_ok=True)


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


def hierarchical_inference(
    tokenizer, dataset, selected_layers=None, n_thought_path=1, detailed=False, temperature=0.5, path_correct=False
):
    # soft_decision = True
    soft_decision = n_thought_path > 1
    HI = True if selected_layers else False

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

    row = arch[:-2] + (' HI' if HI else ' ZS') + '     &' + '&'.join(['{:^9}'] * len(layer_cnt)) + '\\\\'

    for i, texts in enumerate(text_inputs):
        print(f'for prompts {i}#')

        text_features = encode_text_batch(
            model, tokenizer, texts=texts, n_classes=layer_offset[-1], text_fusion=False, device=device
        )

        if not HI:
            selected_layers = [list(range(len(layer_cnt)))]

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
                # similarity = (100.0 * logits).softmax(dim=-1)
                # similarity = (temperature * logits).softmax(dim=-1)
                similarity = logits
                similarity += 1e-8  # for logic consistency, considered all zero logits
                # for path, target, _logits, in zip(paths, targets, logits):
                #     sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))

                # thinking in hierarchy
                if HI:
                    for i, selected_layer in enumerate(selected_layers):
                        consistent = True  # for soft decision only
                        topk, predicts, last_pred = None, None, None  # satisfies the code analyzer
                        for layer in selected_layer:
                            if layer == selected_layer[0]:  # the first thinking at 1st layer
                                mask = layer_mask[layer]
                            else:  # thoughts based on last thought
                                if soft_decision:  # soft decision have multiple path of thoughts
                                    # TODO: score scaling/processing
                                    # 归一化topk.values 或者该层的logits，温度可以由均值决定。
                                    # scores = torch.softmax(topk.values / topk.values.mean(), dim=-1)
                                    scores = (topk.values + 1e-8) / topk.values.mean()  # 不除以均值，一致性严重降低，和直接最后一层预测没啥区别
                                    scores = torch.softmax(scores * temperature, dim=-1)  # 小值尖锐，大值平滑
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

                            preds[i][layer].append(predicts[0].item() - layer_offset[layer])
                            labels[i][layer].append(targets[layer].item() - layer_offset[layer])

                        # compute hiearchical consistency for soft decision
                        consistency[i] += consistent

                else:
                    consistent = True
                    for layer in selected_layers[0]:
                        masked_similarity = similarity * layer_mask[layer]
                        predicts = masked_similarity.topk(1, dim=-1).indices
                        correct[0][layer] += sum(torch.eq(targets[layer], predicts)).item()

                        if layer != selected_layers[0][0]:  # 非第一层
                            consistent = isa_mask[last_pred.item()][predicts.item()]
                        preds[0][layer].append(predicts.item() - layer_offset[layer])
                        labels[0][layer].append(targets[layer].item() - layer_offset[layer])
                        last_pred = predicts
                    consistency[i] += consistent

        end_time = timeit.default_timer()
        print(f'{dataset.dataset_name} hiearchical inference time: {end_time - start_time :.2f}s')

        # recover Path Correct
        # 对preds进行路径矫正重新和 labels 计算准确率和一致性
        if selected_layers and soft_decision and path_correct:
            for i, selected_layer in enumerate(selected_layers):
                selected_layer = selected_layer[::-1]  # 自下而上更新
                last_pred_layer = selected_layer[0]
                for sample in range(len(dataset)):
                    last_pred = preds[i][last_pred_layer][sample] + layer_offset[last_pred_layer]  # gloabl index
                    for layer in selected_layer[1:]:
                        preds[i][layer][sample] = h.get_node(id=last_pred).get_path()[0][layer].inlayer_idx  # correction
                        last_pred = preds[i][layer][sample] + layer_offset[layer]  # gloabl index
                for layer in selected_layer:
                    correct[i][layer] = sum(np.equal(preds[i][layer], labels[i][layer]))
                consistency[i] = len(dataset)

        ## 评估错误严重程度，用最近公共祖先高度
        # 最深层级版本
        mistake_severity = [[] for _ in selected_layers]
        for i, selected_layer in enumerate(selected_layers):
            species_layer = selected_layer[-1]
            for sample in range(len(preds[i][species_layer])):
                pred = preds[i][species_layer][sample] + layer_offset[species_layer]
                label = labels[i][species_layer][sample] + layer_offset[species_layer]
                # lca = h.get_LCA(pred, label)  # TODO: get_LCA
                # height = h.n_layer - lca.layer if lca else h.n_layer + 1
                if pred == label:
                    height = 1
                    continue  # 跳过正确样本
                else:
                    for node in h.get_node(id=pred).get_path()[0][::-1]:
                        if isa_mask[node.id][label]:
                            height = h.n_layer - node.layer
                            break
                        else:
                            height = h.n_layer + 1  # 没找到，LCA为虚拟根节点
                mistake_severity[i].append(height)

        # 各个层级版本
        # mistake_severity = [[[] for _ in range(len(layer_cnt))] for _ in range(len(selected_layers))]
        # for i, selected_layer in enumerate(selected_layers):
        #     for sample in range(len(preds[i][species_layer])):
        #         for layer in selected_layer:
        #             pred = preds[i][layer][sample] + layer_offset[layer]
        #             label = labels[i][layer][sample] + layer_offset[layer]
        #             # lca = h.get_LCA(pred, label)  # TODO: get_LCA
        #             # height = h.n_layer - lca.layer if lca else h.n_layer + 1
        #             if pred == label:
        #                 height = 1
        #                 continue  # 跳过正确样本
        #             else:
        #                 for node in h.get_node(id=pred).get_path()[0][::-1]:
        #                     if isa_mask[node.id][label]:
        #                         height = h.n_layer - node.layer  # 应不应该用相对高度（可以不用）
        #                         break
        #                 else:
        #                     height = h.n_layer + 1
        #             mistake_severity[i][layer].append(height)

        # latex accuracy output
        # if HI:
        for i, selected_layer in enumerate(selected_layers):
            res = [
                '{:.2f}'.format(correct[i][idx] * 100 / len(dataset)) if idx in selected_layer else '-'
                for idx in range(len(layer_cnt))
            ]
            consistency_report = f'consistency: {consistency[i] * 100 / len(dataset): .2f}'  # if soft_decision else ''
            mistake_severity_report = f'mistake serverity: {np.mean(mistake_severity[i]) / h.n_layer * 100:.2f}'
            print(f'{row.format(*res)} {consistency_report:>25} {mistake_severity_report:>27}')
        # else:
        # res = ['{:.2f}'.format(correct[idx] * 100 / len(dataset)) for idx in range(len(layer_cnt))]
        # print(row.format(*res))

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

    clip_transform = Compose([torch.HalfTensor])
    clip_target_transform = Compose([torch.as_tensor])

    collect_image_features(model)
    # collect_clip_logits(tokenizer, text_fusion=True, dump=False)
    # collect_clip_logits(tokenizer, text_fusion=True, only=3, dump=False)

    def test_iwildcam36():
        dataset = get_feature_dataset(dataset_name='iwildcam36', split='test', model=model_name, arch=arch)
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(tokenizer, dataset)

        print('hard decision')
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

        print('soft decision')
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
            n_thought_path=1000,
        )

        print('path correct')
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
            n_thought_path=1000,
            path_correct=True,
        )

    def iwildcam36_temperature():
        dataset = get_feature_dataset(dataset_name='iwildcam36', split='test', model=model_name, arch=arch)
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(tokenizer, dataset)

        for t in range(1, 10):
            t = 1 / t
            # t = t / 10
            print(f'temperature: {t}')
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
                n_thought_path=1000,
                temperature=t,
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

    test_iwildcam36()
    # test_aircraft()
    # test_animal90()


def benchmark_torchsave_pickledump():
    # comparsion between torch.load vs. pickle.load
    # torch loading time:  16.87s
    # pickle loading time:  3.20s

    start_time = timeit.default_timer()
    data = torch.load('work/feature_dataset/clip/ViT-B-32_pre/imagenet1k_val_features.pth')
    data = torch.load('work/feature_dataset/clip/ViT-B-32_pre/imagenet1k_train_features.pth')
    end_time = timeit.default_timer()
    print(f'torch loading time: {end_time - start_time :.2f}s')

    start_time = timeit.default_timer()
    with open('work/feature_dataset/clip/ViT-B-32/imagenet1k_val_features.pkl', 'rb') as file:
        data = pickle.load(file)
    with open('work/feature_dataset/clip/ViT-B-32/imagenet1k_train_features.pkl', 'rb') as file:
        data = pickle.load(file)
    end_time = timeit.default_timer()
    assert data
    print(f'pickle loading time: {end_time - start_time :.2f}s')
