import os
import pandas as pd
from torch import layer_norm
from hierarchical.hierarchy import Hierarchy, get_hierarchy

from dataset.imagenet1k import imagenet_classes, imagenet_templates, imagenet_templates_subset

output_dir = '/root/projects/readings/work/prompts'
os.makedirs(output_dir, exist_ok=True)


def imagenet_prompts(classnames):
    """from clsnames to prompt texts

    Args:
        clsnames (Iterable): text label(s) for each class

    Returns:
        prompt2target (List[Dict]): maps of text prompt and target index, each map represents
        one kind prompt
    """
    prompt2target = []

    # prompt 0
    texts = [f'a photo of a {c[0]}' for c in classnames]
    # m = {t: i for i, t in enumerate(texts)}
    prompt2target.append(texts)

    # prompt 1
    texts = [f'a photo of a {", ".join(c)}' for c in classnames]
    prompt2target.append(texts)

    # prompt 2
    texts = [f'a photo of a {", or ".join(c)}' for c in classnames]
    prompt2target.append(texts)

    # prompt 3
    texts = [f'a photo of a {c}' for c in imagenet_classes]
    prompt2target.append(texts)

    # prompt 4
    texts = [template.format(c) for c in imagenet_classes for template in imagenet_templates_subset]
    prompt2target.append(texts)

    # prompt 5
    texts = [template.format(c) for template in imagenet_classes for c in imagenet_templates]
    prompt2target.append(texts)

    return prompt2target


# def iwildcam36_prompts(h: Hierarchy):
#     prompts = []
#     layer_cnt = [0] * (h.n_layer + 1)
#     for i in range(len(h.nodes)):
#         node = h.get_node(i)
#         prompts.append(node.prompt('a photo of a {}'))
#         layer_cnt[node.layer] += 1
#     layer_cnt.append(sum(layer_cnt))
#     return [prompts], layer_cnt


def iwildcam36_prompts(h: Hierarchy):
    """return prompt for each class in the hierarchy.

    Args:
        h (Hierarchy): hierarchy object for the hierarchical label set.

    Returns:
        [prompts], pointers, layer_cnt (Tuple[List[List[str]], List[List[Int]], List[Int]]):
            group of prompts, pointers[idx][layer] = descendants, cnts of each layer.
    """
    prompts = []
    pointers = []
    layer_cnt = []
    roots = h.get_roots()

    while len(roots) != 0:
        todo = []
        roots.sort(key=lambda node: node.inlayer_idx)
        for root in roots:
            todo.extend(root.children)
            if root.is_leaf():
                prompts.append(f'a photo of a {root.english_name}, a kind of {root.get_root().name} , in the wild.')
            elif root.is_root():
                # prompts.append(f'a photo of a {root.name}.')

                # describe next layer.
                # desp = ', or '.join(c.name for c in root.children)
                desp = ', '.join(c.name for c in root.children)
                prompts.append(f'a photo of a {root.name} such as {desp}.')
            else:
                # describe next layer.
                desp = ', '.join(c.name for c in root.children)
                # describe prev layer.
                parent = root.parents[0]
                prompts.append(f'a photo of a {root.name}, a kind of {parent.name}, such as {desp}.')

            # convert List[List[Node]] to List[List[global index]]
            descendants = root.hierarchical_descendants()
            pointer = [[node.id for node in layer] for layer in descendants]
            assert len(pointer) == len(descendants)
            pointers.append(pointer)

        layer_cnt.append(len(roots))
        roots = todo

    return [prompts], pointers, layer_cnt


def hierarchical_prompt(h: Hierarchy):
    return {
        'imagenet1k': None,
        'iwildcam36': iwildcam36_prompts(h),
    }[h.dataset]


def clsname2prompt(dataset, classnames):
    return {
        'imagenet1k': imagenet_prompts,
    }[
        dataset
    ](classnames)


if __name__ == '__main__':
    h = get_hierarchy('iwildcam36')
    prompts, layer_cnt = iwildcam36_prompts(h)
    prompts2, layer_cnt2 = iwildcam36_prompts(h)

    print(prompts == prompts2)
    print(layer_cnt == layer_cnt2)
