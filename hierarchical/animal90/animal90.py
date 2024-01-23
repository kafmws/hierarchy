import os
import sys

curdir = os.path.abspath(os.curdir)
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

import pandas as pd


animal_classnames = [
    "antelope",
    "badger",
    "bat",
    "bear",
    "bee",
    "beetle",
    "bison",
    "boar",
    "butterfly",
    "cat",
    "caterpillar",
    "chimpanzee",
    "cockroach",
    "cow",
    "coyote",
    "crab",
    "crow",
    "deer",
    "dog",
    "dolphin",
    "donkey",
    "dragonfly",
    "duck",
    "eagle",
    "elephant",
    "flamingo",
    "fly",
    "fox",
    "goat",
    "goldfish",
    "goose",
    "gorilla",
    "grasshopper",
    "hamster",
    "hare",
    "hedgehog",
    "hippopotamus",
    "hornbill",
    "horse",
    "hummingbird",
    "hyena",
    "jellyfish",
    "kangaroo",
    "koala",
    "ladybugs",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "mosquito",
    "moth",
    "mouse",
    "octopus",
    "okapi",
    "orangutan",
    "otter",
    "owl",
    "ox",
    "oyster",
    "panda",
    "parrot",
    "pelecaniformes",
    "penguin",
    "pig",
    "pigeon",
    "porcupine",
    "possum",
    "raccoon",
    "rat",
    "reindeer",
    "rhinoceros",
    "sandpiper",
    "seahorse",
    "seal",
    "shark",
    "sheep",
    "snake",
    "sparrow",
    "squid",
    "squirrel",
    "starfish",
    "swan",
    "tiger",
    "turkey",
    "turtle",
    "whale",
    "wolf",
    "wombat",
    "woodpecker",
    "zebra",
]

from hierarchy import Hierarchy, Node, Attributes


class AnimalNode(Node):
    """parents are all attributes"""

    def __init__(self, id: str | int, name: str, inlayer_idx: int, layer: int, layername: str):
        """class definition in hierarchy.

        Args:
            id (str | int): global index of all classes in the hierarchy.
            name (str): class name.
            inlayer_idx (int): index of current layer of the hierarchy.
            layer (int): layer index, for `layer=4` (species), node owns
                additional english name and description.
            layername (str): layer name, refer```
                    layermap = {
                        'habitat': 0,
                        'class': 1,
                        'english name': 2
                    }
                ```.
        """
        super().__init__(id, name.lower(), layer)
        self.inlayer_idx = inlayer_idx
        self.layername = layername


meta = pd.read_csv('taxonomy.csv')
h = Hierarchy(node_class=AnimalNode, dataset='animal90', htype='attr')
attr_names = ['habitat', 'class']
class_names = ['english name']
h.layermap = {}

# define attributes
name2id = {}
h.n_layer = len(attr_names) + len(class_names)  # 3
h.layer_cnt = []
for layer, attr_name in enumerate(attr_names):
    attr_group = Attributes(attr_name)
    attrs = meta[attr_name].unique().tolist()
    for idx, attr in enumerate(attrs):
        attr_node = h.get_node(id=len(h), name=attr, inlayer_idx=idx, layer=layer, layername=attr_name)
        attr_group.add_attribute(attr_node)
        name2id[attr] = len(name2id)
    h.add_attributes(attr_group)
    h.layer_cnt.append(len(attr_group))
    h.layermap[attr_name] = len(h.layermap)


# define classes and related attributes
for keyword in class_names:
    layer = len(h.layermap)
    for i in range(len(meta)):
        speices = meta.iloc[i][keyword]
        node = h.get_node(id=len(h), name=speices, inlayer_idx=i, layer=layer, layername=keyword)
        for attr_name in attr_names:
            attr = meta.iloc[i][attr_name]
            attr_node = h.get_node(id=name2id[attr], name=attr)
            h.add_edge(attr_node, node)
    h.layer_cnt.append(len(attr_group))
    h.layermap[keyword] = len(h.layermap)

os.chdir(curdir)
