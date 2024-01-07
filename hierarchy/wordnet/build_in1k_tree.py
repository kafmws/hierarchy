import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Counter, Dict, Iterable, List, Set, Tuple
from typing import Callable
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import seaborn as sns


# list all imagenet 1k ids
in1k: Set[str] = set(os.listdir('/data/imagenet_sets/in1k/train/'))


# map nid to classnames
id2name = {}
with open("words.txt") as f:
    for line in f:
        wnid, words = line.strip().split(maxsplit=1)
        id2name[wnid] = words

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
        return self.id + ' ' + id2name[self.id]
    
    def __str__(self):
        return self.name
    
    def printHiearchy(self):
        print([str(p) for p in self.parents])
        print(self.id)
        print([str(p) for p in self.children])


nodes: Dict[str, Node] = {}  # tree["id"] = node


def getNode(id: str) -> Node:
    if id not in nodes:
        nodes[id] = Node(id)
    node = nodes[id]
    return node


# extract imagenet1k tree from is_a.txt, imagenet1k classes are leaf nodes
curlayer: Set[str] = in1k.copy()  # current layer ids
_flags: Set[str] = in1k.copy()    # searched ids
with open('is_a.txt', 'r') as f:
    pc_pairs = [(line.strip().split(' ')) for line in f.readlines()]
    while len(curlayer):
        nextlayer = set()
        for parent, child in pc_pairs:
            if parent in in1k:  # imagenet1k classes are leaf nodes
                continue
            pnew, cnew = parent in curlayer, child in curlayer
            if pnew or cnew:  # next layer
                parentNode, childNode = getNode(parent), getNode(child)
                parentNode.addChild(childNode)
                childNode.addParent(parentNode)
                if not pnew and parent not in _flags:
                    nextlayer.add(parent)
                if not cnew and child not in _flags:
                    nextlayer.add(child)
        _flags.update(curlayer)
        curlayer = nextlayer
                    

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

roots = []
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


# build_class_forest(json_name='forest_1k.json')


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
                func.__call__(root)
            
    print(f'{len(deep)} layers', deep)
    return deep, layer2node


# deep, layer2node = layer_travese(roots)
# print(f'all node in the forest: {sum(deep)}')


# test path2leaf
for path in path2leaf('n02569631'):
    print(", ".join(map(str, path)))


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
    print('all imagenet1k classes are wordnet leaf nodes')
print(notleaf)

# 许多 imagenet1k 类别不为叶结点，再往下为细粒度分类（如物种分类）且缺少标注。
# 令 imagenet1k 类别为叶结点，找到最近公共父结点
