import os
import torch
import numpy as np
import pandas as pd
from hierarchical.hierarchy import Hierarchy, get_hierarchy, Node

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
    """return prompt for each class in the hierarchy. iwildcam36 hierarchy is of tree structure.

    Args:
        h (Hierarchy): hierarchy object for the hierarchical label set.

    Returns:
        prompts (List[List[str]):  group of prompts.
        pointers (List[List[torch.LongTensor]]): pointers[cls_idx][layer] = index of descendants.
        layer_mask (np.ndarray): layer_mask[layer] = mask of specified layer for directly caculate result on the layer.
        hierarchical_targets (List[List[Int]]): hierarchical_targets[leaves_cls_idx][layer] = targets of each layer of the leaf classes.
        classnames (List[List[str]]): classnames of each layer.
        h (Hierarchy): label hierarchy of the dataset.
    """
    prompts = []
    pointers = []
    layer_cnt = h.layer_cnt

    roots = h.get_roots()
    while len(roots) != 0:
        todo = []
        roots.sort(key=lambda node: node.inlayer_idx)
        for root in roots:
            todo.extend(root.children)

            """vanilla template with scientific names"""
            # prompts.append(f'a photo of a {root.name}.')

            """english species name"""
            # prompts.append(f'a photo of a {root.english_name if root.is_leaf() else root.name}.')

            """ends with 'in the wild'"""
            # prompts.append(f'a photo of a {(root.english_name + " in the wild") if root.is_leaf() else root.name}.')

            """EVA-CLIP prompts"""
            # prompts.append(f'a {(root.english_name + " in the wild") if root.is_leaf() else root.name}.')

            # """hiearchical prompt"""
            if root.is_leaf():
                # prompts.append(f'a photo of a {root.english_name}, a kind of {root.get_root().name}, in the wild.') # 55.67 & 56.00

                # # prompt engineering baseline
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of animal.')

                # EVA-CLIP prompt
                prompts.append(f'a photo of a {root.english_name}.')

                # # best
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of animal in the wild.')  # 56.86 & 57.14

                # new
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of {root.get_root().name}, a kind of animal.')
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of animal.') # 56.36
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of wildlife.') # 56.28
                # prompts.append(f'a photo of a {root.english_name}, a kind of {root.get_root().name}.') # 52.83
            elif root.is_root():
                # # describe next layer. (best)
                # desp = ', '.join(c.name for c in root.children)
                # prompts.append(f'a photo of a {root.name} such as {desp}.')

                # # prompt engineering baseline
                # prompts.append(f'a camera trap image of a {root.name}, a kind of animal.')

                # EVA-CLIP prompt
                prompts.append(f'a photo of a {root.name}.')

                # new
                # desp = ', '.join(c.name for c in root.children)
                # prompts.append(f'a camera trap image of a {root.name}, a kind of animal, such as {desp}.')
                # prompts.append(f'a camera trap image of a {root.english_name}, a kind of animal.')

            else:
                # # describe next layer. (best)
                # desp = ', '.join(c.name for c in root.children)
                # # describe prev layer.
                # parent = root.parents[0]
                # prompts.append(f'a photo of a {root.name}, a kind of {parent.name}, such as {desp}.')

                # # prompt engineering baseline
                # prompts.append(f'a camera trap image of a {root.name}, a kind of animal.')

                # EVA-CLIP prompt
                prompts.append(f'a photo of a {root.name}.')

                # new
                # prompts.append(f'a camera trap image of a {root.name}, a kind of {parent.name}, such as {desp}.')

            # convert List[List[Node]] to List[List[global index]]
            descendants = root.hierarchical_descendants()
            pointer = [torch.LongTensor([node.id for node in layer]) for layer in descendants]
            assert len(pointer) == len(descendants)
            pointers.append(pointer)
        roots = todo

    isa_mask = []
    layer_mask = []
    classnames = []
    layer_isa_mask = []
    hierarchy_size = len(h)
    _idx = list(range(hierarchy_size))
    for cnt in layer_cnt:
        mask = np.zeros(hierarchy_size)
        mask[_idx[:cnt]] = 1
        layer_mask.append(mask)
        classnames.append([h.get_node(id=idx).name for idx in _idx[:cnt]])
        _idx = _idx[cnt:]
    for node in range(hierarchy_size):
        layer_isa_mask.append([])
        global_mask = np.zeros(hierarchy_size)
        for layer in range(len(pointers[node])):
            mask = np.zeros(hierarchy_size)
            if len(pointers[node][layer]):
                mask[pointers[node][layer]] = 1
            layer_isa_mask[node].append(mask)
            global_mask += mask
        isa_mask.append(global_mask)
    isa_mask = np.array(isa_mask)
    layer_mask = np.array(layer_mask)
    layer_isa_mask = np.array(layer_isa_mask)

    leaves = h.get_leaves()
    leaves.sort(key=lambda node: node.inlayer_idx)
    # print([leaf.inlayer_idx for leaf in leaves])  # check node order
    hierarchical_targets = []
    for leaf in leaves:
        layer_targets = [[] for _ in range(h.n_layer)]
        for node_path in leaf.get_path():
            for layer, node in enumerate(node_path):
                layer_targets[layer].append(node.id)  # global_idx
        hierarchical_targets.append(layer_targets)

    masks = (
        isa_mask,
        layer_isa_mask,
        layer_mask,
    )
    return [prompts], masks, hierarchical_targets, classnames, h


def animal_prompts(h: Hierarchy):
    """return prompt for each class in the hierarchy. hierarchy of animal90 is of attribute structure.
    Note that the options in layer 2 are determined by the combination of attribute layer 0 and 1.

    Args:
        h (Hierarchy): hierarchy object for the hierarchical label set.

    Returns:
        prompts (List[List[str]):  group of prompts.
        pointers (List[List[torch.LongTensor]]): pointers[cls_idx][layer] = index of descendants.
        layer_mask (np.ndarray): layer_mask[layer] = mask of specified layer for directly caculate result on the layer.
        hierarchical_targets (List[List[Int]]): hierarchical_targets[leaves_cls_idx][layer] = targets of each layer of the leaf classes.
        classnames (List[List[str]]): classnames of each layer.
        h (Hierarchy): label hierarchy of the dataset.
    """
    prompts = []
    pointers = []
    layer_cnt = h.layer_cnt

    for layer in range(h.n_layer):
        nodes = h.get_layer_nodes(layer)

        for node in nodes:
            # ViT-L/14@336 ZS     &  81.63  &  86.91  &  97.91  \\
            # if layer == 0 and node.name[0] in ['a', 'e', 'i', 'o', 'u']:
            #     prompts.append(f'a photo of an {node.name}.')
            # else:
            #     prompts.append(f'a photo of a {node.name}.')

            # ViT-L/14@336 ZS     &  85.22  &  88.39  &  97.91  \\
            if layer == 0:
                if node.name[0] in ['a', 'e', 'i', 'o', 'u']:
                    prompts.append(f'a photo of an {node.name} animal.')
                else:
                    prompts.append(f'a photo of a {node.name} animal.')
            else:
                if node.name[0] in ['a', 'e', 'i', 'o', 'u']:
                    prompts.append(f'a photo of an {node.name}.')
                else:
                    prompts.append(f'a photo of a {node.name}.')

            # ViT-L/14@336 ZS     &  83.00  &  86.91  &  97.91  \\
            # prompts.append(f'a photo of a {node.name}.')

            # convert List[List[Node]] to List[List[global index]]
            descendants = node.hierarchical_descendants()
            if len(descendants) < h.n_layer:  # align to the layers for pointers
                descendants = [[] * (h.n_layer - len(descendants))] + descendants
            pointer = [torch.LongTensor([node.id for node in layer]) for layer in descendants]
            assert len(pointer) == len(descendants)
            pointers.append(pointer)

    isa_mask = []
    layer_mask = []
    classnames = []
    layer_isa_mask = []
    hierarchy_size = len(h)
    _idx = list(range(hierarchy_size))
    for cnt in layer_cnt:
        mask = np.zeros(hierarchy_size)
        mask[_idx[:cnt]] = 1
        layer_mask.append(mask)
        classnames.append([h.get_node(id=idx).name for idx in _idx[:cnt]])
        _idx = _idx[cnt:]
    for node in range(hierarchy_size):
        layer_isa_mask.append([])
        global_mask = np.zeros(hierarchy_size)
        for layer in range(len(pointers[node])):
            mask = np.zeros(hierarchy_size)
            if len(pointers[node][layer]):
                mask[pointers[node][layer]] = 1
            layer_isa_mask[node].append(mask)
            global_mask += mask
        isa_mask.append(global_mask)
    isa_mask = np.array(isa_mask)
    layer_mask = np.array(layer_mask)
    layer_isa_mask = np.array(layer_isa_mask)

    # layer_mask = []
    # classnames = []
    # hierarchy_size = len(h)
    # _idx = list(range(hierarchy_size))
    # for cnt in layer_cnt:
    #     mask = np.zeros((1, hierarchy_size))
    #     mask[0][_idx[:cnt]] = 1
    #     layer_mask.append(mask)
    #     classnames.append([h.get_node(id=idx).name for idx in _idx[:cnt]])
    #     _idx = _idx[cnt:]
    # for node in range(hierarchy_size):
    #     for layer in range(len(pointers[node])):
    #         if len(pointers[node][layer]):
    #             mask = torch.zeros(1, hierarchy_size)
    #             mask[0][pointers[node][layer]] = 1
    #             pointers[node][layer] = mask
    # layer_mask = np.array(layer_mask)

    leaves = h.get_leaves()
    leaves.sort(key=lambda node: node.inlayer_idx)
    # print([leaf.inlayer_idx for leaf in leaves])  # check node order
    hierarchical_targets = []
    for leaf in leaves:
        layer_targets = [[] for _ in range(h.n_layer)]
        layer_targets[-1].append(leaf.id)  # add leaf index at leaf layer
        for node_path in leaf.get_path():
            for layer, node in enumerate(node_path[:-1]):  # drop leaf layer for excluding repeated leaf node
                layer_targets[node.layer].append(node.id)  # global_idx
        hierarchical_targets.append(layer_targets)

    masks = (
        isa_mask,
        layer_isa_mask,
        layer_mask,
    )
    return [prompts], masks, hierarchical_targets, classnames, h


def aircraft_prompts(h: Hierarchy):
    """return prompt for each class in the hierarchy. fgvc_aircraft hierarchy is of tree structure.

    Args:
        h (Hierarchy): hierarchy object for the hierarchical label set.

    Returns:
        prompts (List[List[str]):  group of prompts.
        pointers (List[List[torch.LongTensor]]): pointers[cls_idx][layer] = index of descendants.
        layer_mask (np.ndarray): layer_mask[layer] = mask of specified layer for directly caculate result on the layer.
        hierarchical_targets (List[List[Int]]): hierarchical_targets[leaves_cls_idx][layer] = targets of each layer of the leaf classes.
        classnames (List[List[str]]): classnames of each layer.
        h (Hierarchy): label hierarchy of the dataset.
    """
    prompts = []
    pointers = []
    layer_cnt = h.layer_cnt

    roots = h.get_roots()
    while len(roots) != 0:
        todo = []
        roots.sort(key=lambda node: node.inlayer_idx)
        for root in roots:
            todo.extend(root.children)
            # if root.is_leaf():
            #     prompts.append(f'a photo of a {root.english_name}, a kind of {root.get_root().name} , in the wild.')
            # elif root.is_root():
            #     # prompts.append(f'a photo of a {root.name}.')

            #     # describe next layer.
            #     # desp = ', or '.join(c.name for c in root.children)
            #     desp = ', '.join(c.name for c in root.children)
            #     prompts.append(f'a photo of a {root.name} such as {desp}.')
            # else:
            #     # describe next layer.
            #     desp = ', '.join(c.name for c in root.children)
            #     # describe prev layer.
            #     parent = root.parents[0]
            #     prompts.append(f'a photo of a {root.name}, a kind of {parent.name}, such as {desp}.')

            # prompt 2
            # if root.name[0].lower() in ['a', 'e', 'i', 'o', 'u']:
            #     prompts.append(f'a photo of an {root.name} aircraft.')
            # else:
            #     prompts.append(f'a photo of a {root.name} aircraft.')

            # prompt 3
            # prompts.append(f'a photo of a {root.name}, a type of aircraft.')

            # prompt 4
            prompts.append(f'a photo of a {root.name}, a type of airplane.')

            # convert List[List[Node]] to List[List[global index]]
            descendants = root.hierarchical_descendants()
            pointer = [torch.LongTensor([node.id for node in layer]) for layer in descendants]
            assert len(pointer) == len(descendants)
            pointers.append(pointer)
        roots = todo

    isa_mask = []
    layer_mask = []
    classnames = []
    layer_isa_mask = []
    hierarchy_size = len(h)
    _idx = list(range(hierarchy_size))
    for cnt in layer_cnt:
        mask = np.zeros(hierarchy_size)
        mask[_idx[:cnt]] = 1
        layer_mask.append(mask)
        classnames.append([h.get_node(id=idx).name for idx in _idx[:cnt]])
        _idx = _idx[cnt:]
    for node in range(hierarchy_size):
        layer_isa_mask.append([])
        global_mask = np.zeros(hierarchy_size)
        for layer in range(len(pointers[node])):
            mask = np.zeros(hierarchy_size)
            if len(pointers[node][layer]):
                mask[pointers[node][layer]] = 1
            layer_isa_mask[node].append(mask)
            global_mask += mask
        isa_mask.append(global_mask)
    isa_mask = np.array(isa_mask)
    layer_mask = np.array(layer_mask)
    layer_isa_mask = np.array(layer_isa_mask)

    leaves = h.get_leaves()
    leaves.sort(key=lambda node: node.inlayer_idx)
    # print([leaf.inlayer_idx for leaf in leaves])  # check node order
    hierarchical_targets = []
    for leaf in leaves:
        layer_targets = [[] for _ in range(h.n_layer)]
        for node_path in leaf.get_path():
            for layer, node in enumerate(node_path):
                layer_targets[layer].append(node.id)  # global_idx
        hierarchical_targets.append(layer_targets)

    masks = (
        isa_mask,
        layer_isa_mask,
        layer_mask,
    )
    return [prompts], masks, hierarchical_targets, classnames, h


def hierarchical_prompt(dataset_name):
    h: Hierarchy = get_hierarchy(dataset_name)

    return {
        'imagenet1k': None,
        'iwildcam36': iwildcam36_prompts,
        'animal90': animal_prompts,
        'aircraft': aircraft_prompts,
    }[
        dataset_name
    ](h)


def clsname2prompt(dataset, classnames):
    return {
        'imagenet1k': imagenet_prompts,
    }[
        dataset
    ](classnames)


if __name__ == '__main__':
    # h = get_hierarchy('animal90')
    # prompts, pointers, layer_mask, hierarchical_targets, classnames, h = hierarchical_prompt('animal90')
    prompts, layer_mask, hierarchical_targets, classnames, h = hierarchical_prompt('iwildcam36')
    h: Hierarchy = h

    with open('/root/projects/readings/vis/tree_vis/h.txt', 'w') as htxt, open(
        '/root/projects/readings/vis/tree_vis/decorate.txt', 'w'
    ) as decoratetxt:
        roots = h.get_roots()
        colors = 'rbg'
        colormap = dict(zip(roots, colors))
        print(colormap)

        for leaf in h.get_leaves():
            leaf: Node = leaf
            path = leaf.get_path()

            textpath = ''
            for node in path[0]:
                # if node.is_leaf():
                #     textpath += node.english_name
                # else:
                name = node.name
                name = name[0].upper() + name[1:]
                textpath += name.replace(' ', '_')
                textpath += '.'

                if not node.is_leaf():
                    print(f'{name}	annotation	{name}', file=decoratetxt)
                    if node.layer == 1:
                        print(f'{name}	annotation_background_color	{colormap[node.get_root()]}', file=decoratetxt)

            print(textpath[:-1], file=htxt)
