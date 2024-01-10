import os
from graphviz import Digraph
from typing import Callable, Dict, Set, List, Tuple


class Node:
    
    def __init__(self, id: str, name: str, idx: int = None, layer: int = None):
        """class in hierarchy.

        Args:
            id (str): unique in the hierarchy.
            name (str): description of the class, text label.
            idx (str): class idx of the dataset, default is -1.
        """
        self.id = str(id)
        self.name = name
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        self.idx = idx
        self.layer = layer
        
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
    
    def printHiearchy(self):
        print([str(p) for p in self.parents])
        print(self.id)
        print([str(p) for p in self.children])


class Hierarchy:
    
    def __init__(self, node_class=Node):
        self.nodecls = node_class
        self.nodes: Dict[str, self.nodecls] = {}  # nodes['id'] = node: Node
    
    def __contains__(self, id) -> bool:
        return id in self.nodes

    def getNode(self, id: str, name: str, idx: int = None, layer: int = None) -> Node:
        if id not in self.nodes:
            assert name is not None, 'name should not be None when add new node'
            self.nodes[id] = self.nodecls.__call__(id, name, idx, layer)
        node = self.nodes[id]
        return node
    
    def add_edge(self, p: str | Node, c: str | Node):
        if isinstance(p, str):
            assert p in self
            p = self.getNode(p)
        if isinstance(c, str):
            assert c in self
            c = self.getNode(c)
        p.addChild(c)
        c.addParent(p)
    
    def get_roots(self) -> List[Node]:
        return [node for node in self.nodes.values() if len(node.parents) == 0]
    
    def get_nodes_edges(self, root: Node) -> Tuple[List[Node], List[Tuple[Node, Node]]]:
        """return nodes and edges of subtree of Node `root`, compatible with graph.
        Args:
            root (Node): root node of the tree.
        Returns:
            nodes and edges (tuple): Tuple of list of nodes and list of edges, each edge is tuple(parent, child).
        """
        nodes: Set[Node] = set()
        edges: Set[Tuple(Node, Node)] = set()
        
        cur = [root]
        while len(cur):
            todo: Set[Node] = set()
            for node in cur:
                if node in nodes:
                    continue
                nodes.add(node)
                for child in node.children:
                    todo.add(child)
                    edges.add((node, child))
            cur = todo
        return list(nodes), list(edges)
    
    def _export_graphviz(self, roots: Node | List[Node] = None, dotfile='tree.dot', dotname: Callable = None):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, Node):
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

    def export_graphviz(self, roots: Node | List[Node] = None, dotfile='tree.dot', dotname: Callable = None):
        if roots is None:
            roots = self.get_roots()
        if isinstance(roots, Node):
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
