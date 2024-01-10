import os
import sys
import json

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from hierarchy import Hierarchy

# load hierarchical infomation
with open('tree_desc.json', 'r') as file:
    tree = json.load(file)
    layer_cnt = tree.pop('ext_len_info')
    
with open('cls_to_idx.json', 'r') as file:
    cls_to_idx = json.load(file)
    cls_to_idx = {name.replace('_', ' '): idx for name, idx in cls_to_idx.items()}
    
h = Hierarchy()
for layer, layer_name in enumerate(tree.keys()):
    for parent, children in tree[layer_name].items():
        cls_to_idx.setdefault(parent, len(cls_to_idx))
        p_idx = cls_to_idx[parent]
        p_node = h.getNode(p_idx, parent, p_idx, layer)
        if layer_name == 'info_species':
            p_node.alias = children[0]
            p_node.description = children[1]
        else:
            for child in children:
                cls_to_idx.setdefault(child, len(cls_to_idx))
                c_idx = cls_to_idx[child]
                c_node = h.getNode(c_idx, child, c_idx, layer)
                h.add_edge(p_node, c_node)

h.export_graphviz(dotfile='iwildcam74.dot')
