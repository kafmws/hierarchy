import os
import sys
from graphviz import Digraph
from typing import Callable, Dict, Generic, Set, List, Tuple, Type, TypeVar

sys.path.append(os.path.dirname(__file__))


class Node:
    def __init__(self, id: str | int, name: str, layer: int = None, inlayer_idx: int = None):
        """class definition in hierarchy.

        Args:
            id (str | int): unique in the hierarchy.
            name (str): description of the class, text label.
            inlayer_idx (str): class idx of the layer of the node.
            layer (int): layer the node belongs to.
        """
        self.id = id
        self.name = name
        self.layer = layer
        self.inlayer_idx = inlayer_idx
        self._parents: Set[Node] = set()
        self._children: Set[Node] = set()

        self.layer_path: List[List[Node]] = None

    def addParent(self, p):
        assert isinstance(p, Node)
        self._parents.add(p)

    def addChild(self, c):
        assert isinstance(c, Node)
        self._children.add(c)

    def __str__(self):
        return str(self.id) + ' ' + self.name

    @property
    def dotid(self):
        return str(self)

    @property
    def dotlabel(self):
        return self.name

    @property
    def parents(self) -> List:
        return sorted(self._parents, key=lambda node: node.id)

    @property
    def children(self) -> List:
        return sorted(self._children, key=lambda node: node.id)

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
        return len(self._parents) == 0

    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def get_root(self):
        if not self.is_root():
            return self.parents[0].get_root()
        return self

    def hierarchical_descendants(self) -> List[List['Node']]:
        """return all descendant nodes at lyaer `layer`.
        Return:
            descendants (List[List[Node]]): descendants[layer] = descendants_at_layer.
        """
        assert self.layer is not None, f'node `{str(self)}.layer` is None!'
        descendants = [[]] * (self.layer + 1)

        cur = self.children
        while len(cur) != 0:
            nextlayer = set()
            for node in cur:
                nextlayer.update(node.children)
            descendants.append(list(cur))
            cur = nextlayer
        return descendants


NodeClass = TypeVar('NodeClass', bound=Node)


class Attributes(Generic[NodeClass]):
    """A group of atrributes of a kind of complete partition of deepest layer classes"""

    def __init__(self, name):
        self.name = name
        self.attrs: List[NodeClass] = []

    def add_attribute(self, node: NodeClass):
        self.attrs.append(node)

    def __len__(self):
        return len(self.attrs)

    def __contains__(self, node: NodeClass | str) -> bool:
        for attr in self.attrs:
            if node == attr or node == attr.name:
                return True
        return False


class Hierarchy(Generic[NodeClass]):
    def __init__(self, node_class: Type[NodeClass], n_layer: int, dataset: str = None, htype='tree'):
        """the representation of the hierachical labels.

        Args:
            node_class (Type[NodeClass]): the class represents classes of the hierarchy.
            n_layer (int): the total layer number of the hierarchy.
            dataset (str, optional): dataset name of the hierarchical labels. Defaults to None.
            htype (str, optional): defined in `['tree', 'attr', 'hybird']`. Defaults to 'tree'.
        """
        self.type = htype
        self.dataset = dataset
        self.n_layer = n_layer
        self.node_class = node_class
        self.layer_cnt = [0] * n_layer  # node count of each layer

        self.nodes: Dict[str, NodeClass] = {}  # nodes['id'] = node: Node
        self.attributes: List[Attributes] = []

    def __contains__(self, node: NodeClass | str) -> bool:
        for id, node in self.nodes.items():
            if node == id or node == node:
                return True
        return False

    def __len__(self) -> int:
        return len(self.nodes)

    def get_node(self, id: str, **kwargs) -> NodeClass:
        if id not in self.nodes:
            assert all(k in kwargs for k in ('name', 'layer')), '`name` and `layer` should not be None when add new node'
            self.nodes[id] = self.node_class(id=id, **kwargs)
            self.layer_cnt[kwargs['layer']] += 1
        node = self.nodes[id]
        return node

    def add_edge(self, p: str | NodeClass, c: str | NodeClass):
        if isinstance(p, str):
            assert p in self
            p = self.get_node(p)
        if isinstance(c, str):
            assert c in self
            c = self.get_node(c)
        p.addChild(c)
        c.addParent(p)

    def add_attributes(self, attr_group: Attributes):
        assert self.type in ['attr', 'hybird'], f'defined hierarchy type {self.type} not support attributes.'
        self.attributes.append(attr_group)

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
        edges: List[Set[Tuple[NodeClass, NodeClass]]] = list()

        cur = set([root])
        while len(cur):
            todo: Set[NodeClass] = set()
            curedges: Set[Tuple[NodeClass, NodeClass]] = set()
            nodes.append(cur)
            for node in cur:
                for child in node.children:
                    todo.add(child)
                    curedges.add((node, child))
            cur = todo
            edges.append(curedges)
        return [list(s) for s in nodes], [list(s) for s in edges]

    def get_layer_nodes(self, layer: int) -> List[NodeClass]:
        assert layer >= 0 and layer < self.n_layer, f'`layer` {layer} must in [0, {self.n_layer}) of the hierarchy.'
        layernodes = [node for id, node in self.nodes.items() if node.layer == layer]
        return sorted(layernodes, key=lambda node: node.inlayer_idx)

    def get_LCA(self, a, b):
        assert 0  # TODO
        if not isinstance(a, Node):
            a = self.get_node(id=a)
        if not isinstance(b, Node):
            b = self.get_node(id=b)

    def _export_graphviz(self, roots: NodeClass | List[NodeClass] = None, dotfile='tree.dot', dotname: Callable = None):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, self.node_class):
            roots = [roots]

        # output dot file
        with open(dotfile, 'w') as file:
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"' ' {\n')

            # foreach root
            for root in roots:
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"' ' {\n')

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

    def export_graphviz_layerh(self, dotfile='tree.dot', roots: Node | List[Node] = None, ranksep: float = 1):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, Node):
            roots = [roots]

        # output dot file
        with open(dotfile, 'w') as file:
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"' ' {\n')

            file.write(f'\n\tranksep={ranksep}\n')

            # foreach root
            for root in roots:
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"' ' {\n')

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

                file.write('\t}' f'\n\t//root {root.dotid} ends\n')

            # style settings
            file.write('\t//style settings\n')
            file.write('\trankdir = TB;\n')
            file.write('}\n')

        os.system(f'dot {dotfile} -T svg -O')

    def export_graphviz(self, dotfile='./tree.dot', roots: Node | List[Node] = None, ranksep: float = 1, scale: float = 0.95):
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
            file.write(f'strict digraph "{dotfile.split(".dot")[0]}"' ' {\n')

            file.write(f'\n\tranksep = {ranksep}\n')

            # foreach root
            for root in roots:
                file.write(f'\t//root {root.dotid} begins\n')
                file.write(f'\tsubgraph "{root.dotid}"' ' {\n')

                nodes, edges = self.get_layer_nodes_edges(root)

                assert len(nodes) == len(
                    edges
                ), f'len(nodes)({len(nodes)}) != len(edges){len(edges)}, note that last edge list is always empty.'
                for layer, (layernodes, layeredges) in enumerate(zip(nodes, edges)):
                    file.write(f'\t\t//subgraph "layer {layer}"' '{\n')
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

                file.write('\t}' f'\n\t//root {root.dotid} ends\n')

            # style settings
            file.write('\t//style settings\n')
            file.write('\trankdir = TB;\n')
            file.write('}\n')

        os.system(f'dot {dotfile} -T svg -O')


def get_hierarchy(dataset: str) -> Hierarchy:
    from iwildcam36.iwildcam36 import h as h_iwildcam
    from animal90.animal90 import h as h_animal
    from aircraft.aircraft import h as h_aircraft

    h = {'iwildcam36': h_iwildcam, 'animal90': h_animal, 'aircraft': h_aircraft}[dataset]
    h.dataset = dataset
    return h


if __name__ == '__main__':
    hierarchy = get_hierarchy('iwildcam36')
    print(hierarchy)
