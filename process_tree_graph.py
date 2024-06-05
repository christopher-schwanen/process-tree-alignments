from typing import Any

import networkx as nx
import numpy as np
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.path import Path
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.process_tree.utils.generic import is_leaf, is_tau_leaf
from pyvis.network import Network


class ProcessTreeGraph(nx.MultiDiGraph):
    def __init__(self, tree: ProcessTree):
        # self.tree = tree
        super().__init__()
        self._build_process_tree_graph(tree)

    def _build_process_tree_graph(self, tree: ProcessTree) -> None:
        self.add_node(0, source=True)
        if is_tau_leaf(tree):
            return

        match tree.operator:
            case Operator.LOOP:
                if len(tree.children) != 2:
                    raise Exception(f"Loop {tree} does not have exactly two children")
                if is_tau_leaf(tree.children[0]):
                    self.nodes.get(0)["sink"] = True
                    self._build_process_tree_subgraph(tree.children[1], 0, 0)
                else:
                    self.add_node(1, sink=True)
                    self._build_process_tree_subgraph(tree.children[0], 0, 1)
                    self._build_process_tree_subgraph(tree.children[1], 1, 0)
            case None | Operator.SEQUENCE | Operator.XOR | Operator.PARALLEL:
                self.add_node(1, sink=True)
                self._build_process_tree_subgraph(tree, 0, 1)
            case _:
                raise Exception(f"Operator {tree.operator} is not supported")

    def build_process_tree_graph_plain(self, tree: ProcessTree) -> None:
        self.add_node(0, source=True)
        self.add_node(1, sink=True)
        self._build_process_tree_subgraph(tree, 0, 1)

    def _build_process_tree_subgraph(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        match tree.operator:
            case None:
                self._build_process_tree_subgraph_leaf(tree, start_node, end_node, iac)
            case Operator.SEQUENCE:
                self._build_process_tree_subgraph_sequence(tree, start_node, end_node, iac)
            case Operator.XOR:
                self._build_process_tree_subgraph_xor(tree, start_node, end_node, iac)
            case Operator.PARALLEL:
                self._build_process_tree_subgraph_parallel(tree, start_node, end_node, iac)
            case Operator.LOOP:
                self._build_process_tree_subgraph_loop(tree, start_node, end_node, iac)
            case _:
                raise Exception(f"Operator {tree.operator} is not supported")

    def _build_process_tree_subgraph_leaf(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        if not is_leaf(tree):
            raise Exception(f"Subtree {tree} is not a leaf")

        self.add_edge(start_node, end_node, label=tree.label, capacity=1./iac, cost=iac if tree.label is not None else 0)

    def _build_process_tree_subgraph_sequence(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        if tree.operator != Operator.SEQUENCE:
            raise Exception(f"Operator {tree.operator} is not a sequence")
        if len(tree.children) == 0:
            raise Exception(f"Sequence {tree} does not have any children")

        if len(tree.children) == 1:
            self._build_process_tree_subgraph(tree.children[0], start_node, end_node, iac)
            return

        right_node = self.number_of_nodes()
        self.add_node(right_node)
        self._build_process_tree_subgraph(tree.children[0], start_node, right_node, iac=iac)
        for i in range(1, len(tree.children)-1):
            left_node = right_node
            right_node = self.number_of_nodes()
            self.add_node(right_node)
            self._build_process_tree_subgraph(tree.children[i], left_node, right_node, iac=iac)
        self._build_process_tree_subgraph(tree.children[-1], right_node, end_node, iac=iac)

    def _build_process_tree_subgraph_xor(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        if tree.operator != Operator.XOR:
            raise Exception(f"Operator {tree.operator} is not an XOR")

        for child in tree.children:
            self._build_process_tree_subgraph(child, start_node, end_node, iac)

    def _build_process_tree_subgraph_parallel(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        if tree.operator != Operator.PARALLEL:
            raise Exception(f"Operator {tree.operator} is not a parallel")

        if 'shuffle' not in self.nodes.get(start_node).keys():
            self.nodes.get(start_node)["shuffle"] = []
            self.nodes.get(start_node)["iac"] = iac
        if 'shuffle' not in self.nodes.get(end_node).keys():
            self.nodes.get(end_node)["shuffle"] = []
            self.nodes.get(end_node)["iac"] = iac
        shuffle_split = []
        shuffle_join = []
        iac *= len(tree.children)
        for child in tree.children:
            spread_node_start = self.number_of_nodes()
            spread_node_end = spread_node_start + 1
            shuffle_split.append((start_node, spread_node_start, 0))
            shuffle_join.append((spread_node_end, end_node, 0))
            self.add_edge(start_node, spread_node_start, label=None, capacity=1./iac, cost=0, shuffle=True)
            self.add_edge(spread_node_end, end_node, label=None, capacity=1./iac, cost=0, shuffle=True)
            self._build_process_tree_subgraph(child, spread_node_start, spread_node_end, iac)
        self.nodes.get(start_node)["shuffle"].append(shuffle_split)
        self.nodes.get(end_node)["shuffle"].append(shuffle_join)

    def _build_process_tree_subgraph_loop(self, tree: ProcessTree, start_node: Any, end_node: Any, iac: int = 1) -> None:
        if tree.operator != Operator.LOOP:
            raise Exception(f"Operator {tree.operator} is not a loop")
        if len(tree.children) != 2:
            raise Exception(f"Loop {tree} does not have exactly two children")

        old_start_node = start_node
        start_node = self.number_of_nodes()
        self.add_edge(old_start_node, start_node, label=None, capacity=1./iac, cost=0)
        old_end_node = end_node
        end_node = self.number_of_nodes()
        self.add_edge(end_node, old_end_node, label=None, capacity=1./iac, cost=0)

        self._build_process_tree_subgraph(tree.children[0], start_node, end_node, iac)
        self._build_process_tree_subgraph(tree.children[1], end_node, start_node, iac)

    def view(self, rotate: bool = True, rad: float = .0, rad_offset: float = 0.0) -> None:
        pos = nx.kamada_kawai_layout(self)
        nx.draw_networkx_nodes(self, pos,
                               node_color=['g' if 'source' in (k := self.nodes.get(n).keys())
                                           else 'r' if 'sink' in k else 'gray'
                                           for n in self.nodes],
                               node_size=250, alpha=1)
        nx.draw_networkx_labels(self, pos, font_size=10, font_color='w', font_family='sans-serif')

        ax = plt.gca()

        miny = np.min(np.asarray(list(pos.values())), axis=0)[1]
        maxy = np.max(np.asarray(list(pos.values())), axis=0)[1]
        h = maxy - miny

        for e in self.edges:
            if e[0] == e[1]:
                data_loc = pos[e[0]]
                v_shift = 0.125 * h
                h_shift = v_shift * 0.5
                path = [
                    # 1
                    data_loc + np.asarray([0, v_shift]),
                    # 4 4 4
                    data_loc + np.asarray([h_shift, v_shift]),
                    data_loc + np.asarray([h_shift, 0]),
                    data_loc,
                    # 4 4 4
                    data_loc + np.asarray([-h_shift, 0]),
                    data_loc + np.asarray([-h_shift, v_shift]),
                    data_loc + np.asarray([0, v_shift]),
                ]
                conn = ConnectionPatch(pos[e[0]], pos[e[1]], coordsA='data', coordsB='data',
                                       arrowstyle="-|>", mutation_scale=20,
                                       color='b' if 'shuffle' in self.edges.get(e).keys() else "0.5",
                                       shrinkA=7.5, shrinkB=7.5,
                                       patchA=None, patchB=None,
                                       connectionstyle=(lambda posA, posB, *args, **kwargs:
                                                        Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])),
                                       )
                xy1 = pos[e[0]] + np.array([-0.75 * h_shift, 0.525 * v_shift])
                xy2 = pos[e[0]] + np.array([0, v_shift])
                xy3 = pos[e[0]] + np.array([0.75 * h_shift, 0.525 * v_shift])
                angle1 = 90.0
                angle2 = 0.0
                angle3 = -90.0
            else:
                conn = ConnectionPatch(pos[e[0]], pos[e[1]], coordsA='data', coordsB='data',
                                       arrowstyle="-|>", mutation_scale=20,
                                       color='b' if 'shuffle' in self.edges.get(e).keys() else "0.5",
                                       shrinkA=7.5, shrinkB=7.5,
                                       patchA=None, patchB=None,
                                       connectionstyle=f"arc3,rad={rad + rad_offset * e[2]}",
                                       )
                pos_0 = ax.transData.transform(pos[e[0]])
                pos_1 = ax.transData.transform(pos[e[1]])
                d_pos = pos_1 - pos_0
                ctrl = 0.5 * pos_0 + 0.5 * pos_1 + (rad + rad_offset * e[2]) * np.array(
                    [d_pos[1], -d_pos[0]])  # np.array([(0, 1), (-1, 0)]) @ d_pos

                def get_bezier_point_and_angle(t: float) -> tuple[np.ndarray, float]:
                    ctrl_0 = (1 - t) * pos_0 + t * ctrl
                    ctrl_1 = t * pos_1 + (1 - t) * ctrl
                    bezier = (1 - t) * ctrl_0 + t * ctrl_1
                    bezier = ax.transData.inverted().transform(bezier)
                    angle = np.rad2deg(np.arctan((ctrl_1[1] - ctrl_0[1]) / (ctrl_1[0] - ctrl_0[0])))
                    return bezier, angle

                xy1, angle1 = get_bezier_point_and_angle(0.3)
                xy2, angle2 = get_bezier_point_and_angle(0.5)
                xy3, angle3 = get_bezier_point_and_angle(0.7)

            ax.add_patch(conn)

            label = self.edges.get(e)['label']
            cost = self.edges.get(e)['cost']
            capacity = self.edges.get(e)['capacity']

            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))

            ax.text(xy1[0], xy1[1], capacity, size=10, color='b', family='sans-serif', weight='normal', alpha=1.0,
                    ha='center', va='center', rotation=angle1 if rotate else 0.0, transform=ax.transData, bbox=bbox,
                    zorder=1)
            ax.text(xy2[0], xy2[1], label, size=10, color='k', family='sans-serif', weight='normal', alpha=1.0,
                    ha='center', va='center', rotation=angle2 if rotate else 0.0, transform=ax.transData, bbox=bbox,
                    zorder=1)
            ax.text(xy3[0], xy3[1], cost, size=10, color='r', family='sans-serif', weight='normal', alpha=1.0,
                    ha='center', va='center', rotation=angle3 if rotate else 0.0, transform=ax.transData, bbox=bbox,
                    zorder=1)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.axis('off')
        plt.show()


def view_process_tree_graph_pyvis(process_tree_graph: nx.MultiDiGraph) -> None:
    net = Network(width='100%', height='100%', directed=True)
    for node in process_tree_graph.nodes:
        shuffle_dict = process_tree_graph.nodes.get(node).get('shuffle')
        title = str(shuffle_dict) if shuffle_dict else None
        net.add_node(node, label=str(node), size=10,
                     title=title,
                     color='green' if 'source' in process_tree_graph.nodes.get(node).keys()
                     else 'red' if 'sink' in process_tree_graph.nodes.get(node).keys()
                     else 'blue' if 'shuffle' in process_tree_graph.nodes.get(node).keys()
                     else 'gray')
    for edge in process_tree_graph.edges:
        net.add_edge(edge[0], edge[1], label=process_tree_graph.edges.get(edge)['label'],
                     width=10*process_tree_graph.edges.get(edge)['capacity'],
                     color='blue' if 'shuffle' in process_tree_graph.edges.get(edge).keys() else 'gray')
    net.show('process_tree_graph.html')

    # Open the HTML file and parse it with BeautifulSoup
    with open('process_tree_graph.html', 'r+') as f:
        soup = BeautifulSoup(f, 'html.parser')

        soup.div['style'] = 'height: 100%; width: 100%;'

        # Write the modified HTML back to the file
        f.seek(0)
        f.write(str(soup))
        f.truncate()
