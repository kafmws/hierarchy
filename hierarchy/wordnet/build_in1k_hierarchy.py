import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Counter, Dict, Iterable, List, Set, Tuple
from typing import Callable
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import seaborn as sns


# map wnid to classnames from wordnet
wnid2name = {}
with open('words.txt', 'r') as f:
    for line in f:
        wnid, words = line.strip().split(maxsplit=1)
        wnid2name[wnid] = words


# map wnid to class index
in1k_wnid2idx = {}
with open('wnid2idx.txt', 'r') as f:
    for line in f:
        idx, wnid = line.strip().split()
        in1k_wnid2idx[wnid] = int(idx)



# list all imagenet 1k wnids
in1k: Set[str] = set(in1k_wnid2idx.keys())


class Node:
    pass


class Node:
    
    def __init__(self, id: str):
        self.id: str = id
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        self.in1kchildren: Set[Node] = set()
        
    def addParent(self, p):
        assert isinstance(p, Node)
        self.parents.add(p)
        
    def addChild(self, c):
        assert isinstance(c, Node)
        self.children.add(c)
        self.addIN1kChild(c.in1kchildren)
        
    def addIN1kChild(self, c: Node | Iterable):
        if isinstance(c, Node):
            self.in1kchildren.add(c)
        else:
            self.in1kchildren.update(c)

    @property
    def name(self):
        return self.id + ' ' + wnid2name[self.id]
    
    def __str__(self):
        return self.name
    
    def to_json(self):
        children = {child.name: child.to_json() for child in self.children}
        if len(children) == 0:
            return None
        return children
    
    def printHiearchy(self):
        print([str(p) for p in self.parents])
        print(self.id)
        print([str(p) for p in self.children])


nodes: Dict[str, Node] = {}  # node['id'] = node: Node


def getNode(id: str) -> Node:
    if id not in nodes:
        nodes[id] = Node(id)
    node = nodes[id]
    return node


# extract imagenet1k hierarchy from is_a.txt, imagenet1k classes are leaf nodes
curlayer: Set[str] = in1k.copy()  # current layer ids
hierarchy: Set[str] = in1k.copy()    # searched ids
with open('is_a.txt', 'r') as f:
    edges = [(line.strip().split(' ')) for line in f.readlines()]
while len(curlayer):
    upperlayer = set()
    for parent, child in edges:
        if parent in in1k:  # imagenet1k classes are leaf nodes
            continue
        if child in curlayer:  # next upper layer
            parentNode, childNode = getNode(parent), getNode(child)
            parentNode.addChild(childNode)
            childNode.addParent(parentNode)
            if parent not in hierarchy:
                upperlayer.add(parent)
    hierarchy.update(curlayer)
    curlayer = upperlayer
                    

print(f'number of classes: {len(nodes)}')
# node = getNode('n08270417')
# node.printHiearchy()


# output path(s) to leaf node
def path2leaf(leaf: str | Node) -> List[List[Node]]:
    if isinstance(leaf, str):
        leaf = getNode(leaf)
    paths: List[List[Node]] = []
    curpaths = [[leaf]]
    while len(curpaths):
        size = len(curpaths)
        for _ in range(0, size):
            path = curpaths.pop(0)
            n_parent = len(path[-1].parents)
            if n_parent == 0:
                paths.append(path)
            else:
                for i, parent in enumerate(path[-1].parents):
                    p = path.copy() if i + 1 == n_parent else path
                    p.append(parent)
                    curpaths.append(p)
    return paths


# find the longest common path, as the root of the minimum tree
def get_in1k_tree_root():
    leaf_path: List[List[Node]] = []
    for leaf in in1k:
        path = path2leaf(leaf)


# analysis
roots: List[Node] = []
parnetsCnt = []
childrenCnt = []
for id, node in nodes.items():
    if len(node.parents) == 0:
        roots.append(node)
    pcnt = len(node.parents)
    ccnt = len(node.children)
    parnetsCnt.append(pcnt)
    childrenCnt.append(ccnt)

pcounter = Counter(parnetsCnt)
ccounter = Counter(childrenCnt)
print('parent count:', pcounter)
print('child count:', ccounter)


# draw distribution
sns.displot(data=parnetsCnt)
plt.yscale('log')
plt.title('number of parents')
plt.savefig('number of parents (1k tree).png', bbox_inches='tight')
plt.tight_layout()
sns.displot(data=childrenCnt)
plt.yscale('log')
plt.title('number of children')
plt.tight_layout()
plt.savefig('number of children (1k tree).png', bbox_inches='tight')


# test roots
# print([str(p) for p in roots])
# roots = ['n09506337', 'n08887013', 'n09536363', 'n00001740', 'n09572425', 'n09345503', 'n09350045', 'n09050730', 'n10172793', 'n09023321', 'n08860123', 'n08747054']
# test to_json
# print(getNode('n08270417').to_json())


def build_class_forest(json_name='forest.json'):
    with open(json_name, 'w') as f:
        forest = {}
        for root in roots:
            forest[root.name] = root.to_json()
        json.dump(forest, f, indent=4)


build_class_forest(json_name='forest_1k.json')


# count number of classes of each layer
def layer_travese(roots: List[Node], func: Callable = None) -> Tuple[List[int], List[List[Node]]]:
    deep = []
    layer2node = []
    roots = roots.copy()
    while roots:
        size = len(roots)
        deep.append(size)
        layer2node.append(roots.copy())
        for _ in range(0, size):
            root = roots.pop(0)
            roots.extend(root.children)
            
            if func:
                func(root)
            
    print(f'{len(deep)} layers', deep)
    return deep, layer2node


# deep, layer2node = layer_travese(roots)
# print(f'all node in the forest: {sum(deep)}')


def leaves_reachable(roots: Node | List[Node], imagenet_only=True) -> Set[Node]:
    if isinstance(roots, Node):
        roots = [roots]
    leaves: Set[Node] = set()
    roots = roots.copy()
    while roots:
        size = len(roots)
        for _ in range(0, size):
            root = roots.pop(0)
            roots.extend(root.children)
            if len(root.children) == 0 and (root.id in in1k or not imagenet_only):
                leaves.add(root)
    return leaves
    
    
# test path2leaf
# for path in path2leaf('n02569631'):
    # print(", ".join(map(str, path)))


print('=========')
print('roots:')
for r in roots:
    print(f'{r} leaves/imagenet classes: {len(leaves_reachable(r, False))}/{len(leaves_reachable(r))}')
print('=========')


# dump all paths for each node
def dump_paths_nodes():
    with open('paths.json', 'w') as f:
        result = {}
        
        multipathcnt = 0
        
        def dump_node(node: Node):
            path_str_list = []
            for path in path2leaf(node):
                path_str_list.append(list(map(str, path[::-1])))
            if len(path_str_list) > 1:
                global multipathcnt
                multipathcnt += 1
            result[node.name] = path_str_list

        deep, layer2node = layer_travese(roots=roots, func=dump_node)
        json.dump(result, f, indent=4)
        with open('layer2node.json', 'w') as ff:
            layer2class = [[node.name for node in layer] for layer in layer2node]
            json.dump(layer2class, ff, indent=4)
        
        print(f'nodes with multiple path: {multipathcnt}')

# 检查兄弟结点之间的子结点集合是否相互包含了，从而导致多个父类的情况，删除冗余的类
# 删除不符合视觉相似性的层级


# dump_paths_nodes()

# check imagenet 1k classes are all leaf nodes
notleaf = []
for nid in in1k:
    node = getNode(nid)
    if len(node.children):
        notleaf.append(node.name)
else:
    print('all imagenet1k classes are leaf nodes')
print(notleaf)

# 许多 imagenet1k 类别不为叶结点，再往下为细粒度分类（如物种分类）且缺少标注。
# 令 imagenet1k 类别为叶结点，找到最近公共父结点


def export_graphviz(roots: Node | List[Node], dotfile='in1k.dot'):
    if isinstance(roots, Node):
        roots = [roots]
    # TODO use roots to export
    
    # output dot file
    with open(dotfile, 'w') as file:
        file.write('strict digraph imagenet_wordnet {\n')
        
        # generate each node
        file.write('\t//node\n')
        bignode = []  # generate subtrees for big nodes
        wnid2nodeid = {}
        for wnid in nodes.keys():
            name = wnid2name[wnid]
            label = name.split(',')[0]
            cls_idx = in1k_wnid2idx[wnid] if wnid in in1k_wnid2idx else -1
            nodeid = f'{wnid} {name} {cls_idx}'
            wnid2nodeid[wnid] = nodeid
            
            # find big node family
            # if len(ch.leaves_reachable(wnid)) > 50:
            #     cls_idx = -2
            #     nodeid = f'family {name} {cls_idx}'
            #     bignode.append(wnid)
            
            file.write(f'\t"{nodeid}" [label="{label}"')
            if wnid in in1k:
                if cls_idx < 0:
                    # imagenet classes are green, big family nodes are red
                    file.write(f' color="{["red", "green"][cls_idx]}"')
            file.write(']\n')
        
        assert len(wnid2nodeid) == len(nodes), f'node number in dot file != node number in nodes: {len(wnid2nodeid)} != {len(nodes)}'
        
        file.write('\t//edge\n')
        
        # from up to down
        # entity = getNode('n00001740')
        
        # from down to up
        for child in nodes.keys():
            for parentNode in getNode(child).parents:
                file.write(f'\t"{wnid2nodeid[parentNode.id]}" -> "{wnid2nodeid[child]}"\n')
        
        file.write('}\n')
    
    os.system(f'dot {dotfile} -T svg -O')


export_graphviz(roots=roots)
