import os
import json
import queue
import numpy as np
import matplotlib.pyplot as plt
from typing import Counter, Dict, List, Set
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import seaborn as sns

id2name = {}
with open("words.txt") as f:
    for line in f:
        wnid, words = line.strip().split(maxsplit=1)
        id2name[wnid] = words


class Node:
    
    def __init__(self, id: str):
        self.id: str = id
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        
    def addParent(self, p):
        assert isinstance(p, Node)
        self.parents.add(p)
        
    def addChild(self, p):
        assert isinstance(p, Node)
        self.children.add(p)

    @property
    def name(self):
        return self.id + ' ' + id2name[self.id]
    
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


nodes: Dict[str, Node] = {}  # tree["id"] = node


def getNode(id: str) -> Node:
    if id not in nodes:
        nodes[id] = Node(id)
    node = nodes[id]
    return node


with open('is_a.txt', 'r') as f:
    for line in f.readlines():
        parent, child = line.strip().split(' ')
        parentNode, childNode = getNode(parent), getNode(child)
        parentNode.addChild(childNode)
        childNode.addParent(parentNode)

print(f'number of classes: {len(nodes)}')
# node = getNode('n08270417')
# node.printHiearchy()

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
plt.savefig('number of parents.png', bbox_inches='tight')
plt.tight_layout()
sns.displot(data=childrenCnt)
plt.yscale('log')
plt.title('number of children')
plt.tight_layout()
plt.savefig('number of children.png', bbox_inches='tight')


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


build_class_forest(json_name='forest.json')


# count number of classes of each layer
def layer_travese(roots: List[Node]):
    deep = []
    roots = roots.copy()
    while roots:
        size = len(roots)
        deep.append(size)
        for i in range(0, size):
            root = roots.pop(0)
            roots.extend(root.children)
    print(f'{len(deep)} layers', deep)
    return deep


deep = layer_travese(roots)
print(f'all node in the forest: {sum(deep)}')
