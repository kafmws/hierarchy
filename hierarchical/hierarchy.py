import os
import sys
from graphviz import Digraph
from typing import Callable, Dict, Generic, Set, List, Tuple, Type, TypeVar, cast

sys.path.append(os.path.dirname(__file__))


class Node:
    
    def __init__(self, id: str | int, name: str, idx: int = None, layer: int = None):
        """class definition in hierarchy.

        Args:
            id (str | int): unique in the hierarchy.
            name (str): description of the class, text label.
            idx (str): class idx of the dataset, default is -1.
            layer (int): layer the node belongs to.
        """
        self.id = id
        self.name = name
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        self.idx = idx
        self.layer = layer
        
        self.layer_path: List[List[Node]] = None
        
    def addParent(self, p):
        assert isinstance(p, Node)
        self.parents.add(p)
        
    def addChild(self, c):
        assert isinstance(c, Node)
        self.children.add(c)
    
    def __str__(self):
        return self.id + ' ' + self.name
    
    @property
    def dotid(self):
        return str(self)
    
    @property
    def dotlabel(self):
        return self.name
    
    def to_json(self) -> Dict[str, Dict] | None:
        children = {child.name: child.to_json() for child in self.children}
        if len(children) == 0:
            return None
        return children
    
    def get_path(self) -> List[List['Node']]:
        if self.layer_path is not None:
            return self.layer_path
        if self.is_root():
            self.layer_path = [[self]]
            return self.layer_path
        
        self.layer_path = []
        for p in self.parents:
            paths = p.get_path()
            for path in paths:
                path = path.copy()
                path.append(self)
                self.layer_path.append(path)
        return self.layer_path
    
    def is_root(self) -> bool:
        return len(self.parents) == 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_root(self):
        if not self.is_root():
            return list(self.parents)[0].get_root()
        return self
    
    def hierarchical_descendants(self) -> List[List['Node']]:
        """return all descendant nodes at lyaer `layer`.
        Return:
            descendants (List[List[Node]]): descendants[layer] = descendants_at_layer.
        """
        assert self.layer, f'node {str(self)}.layer is None!'
        descendants = [[]] * self.layer
        
        cur = self.children
        while len(cur) != 0:
            nextlayer = set()
            for node in cur:
                nextlayer.update(node.children)
            descendants.append(list(cur))
            cur = nextlayer
        return descendants


NodeClass = TypeVar('NodeClass', bound=Node)


class Hierarchy(Generic[NodeClass]):
    
    def __init__(self, node_class: Type[NodeClass], dataset: str = None, htype='tree'):
        self.type = htype
        self.n_layer = None
        self.dataset = dataset
        self.node_class = node_class
        self.nodes: Dict[str, NodeClass] = {}  # nodes['id'] = node: Node
    
    def __contains__(self, id) -> bool:
        return id in self.nodes
    
    def __len__(self) -> int:
        return len(self.nodes)

    def getNode(self, id: str, **kwargs) -> NodeClass:
        if id not in self.nodes:
            assert 'name' in kwargs, 'name should not be None when add new node'
            self.nodes[id] = self.node_class(id, **kwargs)
        node = self.nodes[id]
        return node
    
    def add_edge(self, p: str | NodeClass, c: str | NodeClass):
        if isinstance(p, str):
            assert p in self
            p = self.getNode(p)
        if isinstance(c, str):
            assert c in self
            c = self.getNode(c)
        p.addChild(c)
        c.addParent(p)
    
    def get_roots(self) -> List[NodeClass]:
        return [node for node in self.nodes.values() if node.is_root()]
    
    def get_leaves(self) -> List[NodeClass]:
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_nodes_edges(self, root: NodeClass) -> Tuple[List[NodeClass], List[Tuple[NodeClass, NodeClass]]]:
        """return nodes and edges of subtree of Node `root`, compatible with graph.
        Args:
            root (Node): root node of the tree.
        Returns:
            nodes and edges (tuple): Tuple of list of nodes and list of edges, each edge is tuple(parent, child).
        """
        nodes, edges = self.get_layer_nodes_edges(root)
        return [node for ls in nodes for node in ls], [edge for ls in edges for edge in ls]
    
    def get_layer_nodes_edges(self, root: NodeClass) -> Tuple[List[List[NodeClass]], List[List[Tuple[NodeClass, NodeClass]]]]:
        """return nodes and edges of subtree of Node `root` by layer, compatible with graph.
        Args:
            root (Node): root node of the tree.
        Returns:
            nodes and edges (tuple): Tuple of list of nodes and edges of each layer, each edge is tuple(parent, child), note that last edge list is always empty.
        """
        nodes: List[Set[NodeClass]] = list()
        edges: List[Set[Tuple(NodeClass, NodeClass)]] = list()
        
        cur = set([root])
        while len(cur):
            todo: Set[NodeClass] = set()
            curedges: Set[Tuple(NodeClass, NodeClass)] = set()
            nodes.append(cur)
            for node in cur:
                for child in node.children:
                    todo.add(child)
                    curedges.add((node, child))
            cur = todo
            edges.append(curedges)
        return [list(s) for s in nodes], [list(s) for s in edges]
    
    def _export_graphviz(self, roots: NodeClass | List[NodeClass] = None, dotfile='tree.dot', dotname: Callable = None):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, self.node_class):
            roots = [roots]
        
        # output dot file
        with open(dotfile, 'w') as file:
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"'' {\n')
            
            # foreach root
            for root in roots:
                
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"'' {\n')
                
                nodes, edges = self.get_nodes_edges(root)
                
                # generate each node, from up to down, layer-by-layer
                file.write('\t\t//node\n')
                for node in nodes:
                    file.write(f'\t\t"{node.dotid}" [label="{node.dotlabel}"]\n')
                
                # record each edge, from up to down, layer-by-layer
                file.write('\t\t//edge\n')
                for pnode, cnode in edges:
                    file.write(f'\t\t"{pnode.dotid}" -> "{cnode.dotid}"\n')

                file.write('\t}\n\t//root {root.dotid} ends\n')
                
            # style settings
            file.write('\t//style settings\n')
            file.write('\trankdir = TB;\n')
            file.write('}\n')
        
        os.system(f'dot {dotfile} -T svg -O')

    def export_graphviz_layerh(self,
                               dotfile='tree.dot',
                               roots: Node | List[Node] = None,
                               ranksep: float = 1):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, Node):
            roots = [roots]
        
        # output dot file
        with open(dotfile, 'w') as file:
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"'' {\n')
            
            file.write(f'\n\tranksep={ranksep}\n')
            
            # foreach root
            for root in roots:
                
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"'' {\n')
                
                nodes, edges = self.get_nodes_edges(root)
                
                # generate each node, from up to down, layer-by-layer
                file.write('\t\t//node\n')
                for node in nodes:
                    # file.write(f'\t\t"{node.dotid}" [label="{node.dotlabel}", scale="{pow(scale, node.layer)}"]\n')
                    file.write(f'\t\t"{node.dotid}" [label="{node.dotlabel}", width="{pow(ranksep, node.layer)}"]\n')
                
                # record each edge, from up to down, layer-by-layer
                file.write('\t\t//edge\n')
                for pnode, cnode in edges:
                    file.write(f'\t\t"{pnode.dotid}" -> "{cnode.dotid}"\n')

                file.write('\t}'f'\n\t//root {root.dotid} ends\n')
                
            # style settings
            file.write('\t//style settings\n')
            file.write('\trankdir = TB;\n')
            file.write('}\n')
        
        os.system(f'dot {dotfile} -T svg -O')

    def export_graphviz(self,
                        dotfile='tree.dot',
                        roots: Node | List[Node] = None,
                        ranksep: float = 1,
                        scale: float = 0.95):
        """_summary_

        Args:
            dotfile (str, optional): _description_. Defaults to 'tree.dot'.
            roots (Node | List[Node], optional): _description_. Defaults to None.
            dotname (Callable, optional): _description_. Defaults to None.
            scale (float, optional): factory to control node size in each layer, `width=scale**layer`. Defaults to 1.
        """
        
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, Node):
            roots = [roots]
        
        # output dot file
        with open(dotfile, 'w') as file:
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"'' {\n')
            
            file.write(f'\n\tranksep = {ranksep}\n')
            
            # foreach root
            for root in roots:
                
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"'' {\n')
                
                nodes, edges = self.get_layer_nodes_edges(root)
                
                assert len(nodes) == len(edges), f'len(nodes)({len(nodes)}) != len(edges){len(edges)}, note that last edge list is always empty.'
                for layer, (layernodes, layeredges) in enumerate(zip(nodes, edges)):
                    
                    file.write(f'\t\t//subgraph "layer {layer}"''{\n')
                    file.write(f'\t\t\tnode [width="{int(pow(scale, layer))}", fontsize={int(14 * pow(scale, layer))}]\n')
                    # file.write(f'\t\t\tnode [fontsize={14 * pow(scale, layer)}' + (', shape="none"'if layer == 4 else '') + ']\n')
                    
                    # generate each node, from up to down, layer-by-layer
                    file.write('\t\t\t//node\n')
                    for node in layernodes:
                        file.write(f'\t\t\t"{node.dotid}" [label="{node.dotlabel}"]\n')
                
                    # record each edge, from up to down, layer-by-layer
                    file.write('\t\t\t//edge\n')
                    for pnode, cnode in layeredges:
                        file.write(f'\t\t\t"{pnode.dotid}" -> "{cnode.dotid}"\n')
                    
                    file.write('\t\t//}\n')

                file.write('\t}'f'\n\t//root {root.dotid} ends\n')
                
            # style settings
            file.write('\t//style settings\n')
            file.write('\trankdir = TB;\n')
            file.write('}\n')
        
        os.system(f'dot {dotfile} -T svg -O')


def get_hierarchy(dataset: str) -> Hierarchy:
    
    from iwildcam36.iwildcam36 import h as iwildcam
    
    h = {
        'iwildcam36': iwildcam
    }[dataset]
    h.dataset = dataset
    return h


if __name__ == '__main__':
    
    hierarchy = get_hierarchy('iwildcam36')
    print(hierarchy)
