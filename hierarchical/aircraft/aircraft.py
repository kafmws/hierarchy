import os
import sys
import json
from itertools import accumulate

curdir = os.path.abspath(os.curdir)
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from dataset import get_dataset
from hierarchy import Hierarchy, Node

# load hierarchical infomation
dataset = get_dataset(dataset_name='aircraft', split='train', transform=None)

layer_map = {
    'manufacturer': 0,
    'family': 1,
    'variant': 2,
}

h: Hierarchy[Node] = Hierarchy(node_class=Node, n_layer=len(layer_map), dataset='aircraft', htype='tree')

h.layer_map = layer_map
classnames = dataset.layer_classes
h.layer2name = {id: name for name, id in layer_map.items()}
layer_offset = list(accumulate([0] + [len(ls) for ls in classnames]))

for path in dataset.layer_targets:
    for layer in range(len(layer_map) - 1):
        c_layer = layer + 1
        parent, child = int(path[layer]), int(path[c_layer])
        p_node = h.get_node(id=parent + layer_offset[layer], name=classnames[layer][parent], layer=layer, inlayer_idx=parent)
        c_node = h.get_node(id=child + layer_offset[c_layer], name=classnames[c_layer][child], layer=c_layer, inlayer_idx=child)
        h.add_edge(p_node, c_node)

if __name__ == '__main__':
    h.export_graphviz(dotfile='./aircraft.dot', scale=0.8)

os.chdir(curdir)
