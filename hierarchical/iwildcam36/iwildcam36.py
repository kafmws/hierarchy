import os
import sys
import json

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
# print('in iwildcam36.py ' + str(sys.path))

from hierarchy import Hierarchy, Node

# load hierarchical infomation
with open('tree_desc.json', 'r') as file:
    tree = json.load(file)

layermap = {
    'class': 1,
    'order': 2,
    'family': 3,
    'genus': 4,
    'species': 5,
}

layermap.update({id: name for name, id in layermap.items()})


class IWildCamNode(Node):
    
    def __init__(self, id: str, name: str, inlayer_idx: int, layer: int, layername: str):
        """class definition in hierarchy.

        Args:
            id (str): global index of all classes in the hierarchy.
            name (str): class name.
            inlayer_idx (int): index of current layer of the hierarchy.
            layer (int): layer index, for `layer=4` (species), node owns
                additional english name and description.
            layer (str): layer name, refer```
                    layer = {
                        'class': 1,
                        'order': 2,
                        'family': 3,
                        'genus': 4,
                        'species': 5,
                    }
                ```.
        """
        super().__init__(id, name.lower(), id, layer)
        self.english_name = None
        self.inlayer_idx = inlayer_idx
        self.layername = layername
    
    def prompt(self, template: str = None) -> str:
        """return prompt for this node"""
        prompt = template.format(self.name) if template is not None else self.name
        return prompt


h: Hierarchy[IWildCamNode] = Hierarchy(IWildCamNode)
h.layermap = layermap
h.n_layer = len(layermap) // 2
for parent in tree.keys():
    p_glabal_idx = tree[parent]['global_idx']
    p_inlayer_idx = tree[parent]['inlayer_idx']
    p_layername = tree[parent]['layer']
    p_node: IWildCamNode = h.getNode(id=p_glabal_idx,
                                     name=parent,
                                     inlayer_idx=p_inlayer_idx,
                                     layer=layermap[p_layername],
                                     layername=p_layername)
    if p_layername == 'species':
        p_node.english_name = tree[parent]['sub_node'][1].lower()
        p_node.description = tree[parent]['sub_node'][2]
        continue
    for child in tree[parent]['sub_node']:
        c_glabal_idx = tree[child]['global_idx']
        c_inlayer_idx = tree[child]['inlayer_idx']
        c_layername = tree[child]['layer']
        c_node = h.getNode(id=c_glabal_idx,
                           name=child,
                           inlayer_idx=c_inlayer_idx,
                           layer=layermap[c_layername],
                           layername=c_layername)
        h.add_edge(p_node, c_node)

if __name__ == '__main__':
    # h.export_graphviz_layerh(dotfile='iwildcam74.dot', ranksep=1)
    h.export_graphviz(dotfile='iwildcam36.dot', scale=0.8)
