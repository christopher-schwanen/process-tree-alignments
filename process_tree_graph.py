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


def build_process_tree_graph(tree: ProcessTree) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node(0, source=True)
    if is_tau_leaf(tree):
        return graph

    match tree.operator:
        case Operator.LOOP:
            if len(tree.children) != 2:
                raise Exception(f"Loop {tree} does not have exactly two children")
            if is_tau_leaf(tree.children[0]):
                graph.nodes.get(0)["sink"] = True
                graph = _build_process_tree_subgraph(tree.children[1], graph, 0, 0)
            else:
                graph.add_node(1, sink=True)
                graph = _build_process_tree_subgraph(tree.children[0], graph, 0, 1)
                graph = _build_process_tree_subgraph(tree.children[1], graph, 1, 0)
        case None | Operator.SEQUENCE | Operator.XOR | Operator.PARALLEL:
            graph.add_node(1, sink=True)
            graph = _build_process_tree_subgraph(tree, graph, 0, 1)
        case _:
            raise Exception(f"Operator {tree.operator} is not supported")
    return graph


def build_process_tree_graph_plain(tree: ProcessTree) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node(0, source=True)
    graph.add_node(1, sink=True)
    graph = _build_process_tree_subgraph(tree, graph, 0, 1)
    return graph


def _build_process_tree_subgraph(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                 iac: int = 1) -> nx.MultiDiGraph:
    match tree.operator:
        case None:
            graph = _build_process_tree_subgraph_leaf(tree, graph, start_node, end_node, iac)
        case Operator.SEQUENCE:
            graph = _build_process_tree_subgraph_sequence(tree, graph, start_node, end_node, iac)
        case Operator.XOR:
            graph = _build_process_tree_subgraph_xor(tree, graph, start_node, end_node, iac)
        case Operator.PARALLEL:
            graph = _build_process_tree_subgraph_parallel(tree, graph, start_node, end_node, iac)
        case Operator.LOOP:
            graph = _build_process_tree_subgraph_loop(tree, graph, start_node, end_node, iac)
        case _:
            raise Exception(f"Operator {tree.operator} is not supported")
    return graph


def _build_process_tree_subgraph_leaf(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                      iac: int = 1) -> nx.MultiDiGraph:
    if not is_leaf(tree):
        raise Exception(f"Subtree {tree} is not a leaf")

    graph.add_edge(start_node, end_node, label=tree.label, capacity=1./iac, cost=iac if tree.label is not None else 0)
    return graph


def _build_process_tree_subgraph_sequence(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                          iac: int = 1) -> nx.MultiDiGraph:
    if tree.operator != Operator.SEQUENCE:
        raise Exception(f"Operator {tree.operator} is not a sequence")
    if len(tree.children) == 0:
        raise Exception(f"Sequence {tree} does not have any children")

    if len(tree.children) == 1:
        graph = _build_process_tree_subgraph(tree.children[0], graph, start_node, end_node, iac)
        return graph

    right_node = graph.number_of_nodes()
    graph.add_node(right_node)
    graph = _build_process_tree_subgraph(tree.children[0], graph, start_node, right_node, iac=iac)
    for i in range(1, len(tree.children)-1):
        left_node = right_node
        right_node = graph.number_of_nodes()
        graph.add_node(right_node)
        graph = _build_process_tree_subgraph(tree.children[i], graph, left_node, right_node, iac=iac)
    graph = _build_process_tree_subgraph(tree.children[-1], graph, right_node, end_node, iac=iac)
    return graph


def _build_process_tree_subgraph_xor(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                     iac: int = 1) -> nx.MultiDiGraph:
    if tree.operator != Operator.XOR:
        raise Exception(f"Operator {tree.operator} is not an XOR")

    for child in tree.children:
        graph = _build_process_tree_subgraph(child, graph, start_node, end_node, iac)
    return graph


def _build_process_tree_subgraph_parallel(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                          iac: int = 1) -> nx.MultiDiGraph:
    if tree.operator != Operator.PARALLEL:
        raise Exception(f"Operator {tree.operator} is not a parallel")

    if 'shuffle' not in graph.nodes.get(start_node).keys():
        graph.nodes.get(start_node)["shuffle"] = []
        graph.nodes.get(start_node)["iac"] = iac
    if 'shuffle' not in graph.nodes.get(end_node).keys():
        graph.nodes.get(end_node)["shuffle"] = []
        graph.nodes.get(end_node)["iac"] = iac
    shuffle_split = []
    shuffle_join = []
    iac *= len(tree.children)
    for child in tree.children:
        spread_node_start = graph.number_of_nodes()
        spread_node_end = spread_node_start + 1
        shuffle_split.append((start_node, spread_node_start, 0))
        shuffle_join.append((spread_node_end, end_node, 0))
        graph.add_edge(start_node, spread_node_start, label=None, capacity=1./iac, cost=0, shuffle=True)
        graph.add_edge(spread_node_end, end_node, label=None, capacity=1./iac, cost=0, shuffle=True)
        graph = _build_process_tree_subgraph(child, graph, spread_node_start, spread_node_end, iac)
    graph.nodes.get(start_node)["shuffle"].append(shuffle_split)
    graph.nodes.get(end_node)["shuffle"].append(shuffle_join)
    return graph


def _build_process_tree_subgraph_loop(tree: ProcessTree, graph: nx.MultiDiGraph, start_node: Any, end_node: Any,
                                      iac: int = 1) -> nx.MultiDiGraph:
    if tree.operator != Operator.LOOP:
        raise Exception(f"Operator {tree.operator} is not a loop")
    if len(tree.children) != 2:
        raise Exception(f"Loop {tree} does not have exactly two children")

    old_start_node = start_node
    start_node = graph.number_of_nodes()
    graph.add_edge(old_start_node, start_node, label=None, capacity=1./iac, cost=0)
    old_end_node = end_node
    end_node = graph.number_of_nodes()
    graph.add_edge(end_node, old_end_node, label=None, capacity=1./iac, cost=0)

    graph = _build_process_tree_subgraph(tree.children[0], graph, start_node, end_node, iac)
    graph = _build_process_tree_subgraph(tree.children[1], graph, end_node, start_node, iac)
    return graph


def view_process_tree_graph(process_tree_graph: nx.MultiDiGraph, rotate: bool = True, rad: float = .0,
                            rad_offset: float = 0.0) -> None:
    pos = nx.kamada_kawai_layout(process_tree_graph)
    nx.draw_networkx_nodes(process_tree_graph, pos,
                           node_color=['g' if 'source' in (k := process_tree_graph.nodes.get(n).keys())
                                       else 'r' if 'sink' in k else 'gray'
                                       for n in process_tree_graph.nodes],
                           node_size=250, alpha=1)
    nx.draw_networkx_labels(process_tree_graph, pos, font_size=10, font_color='w', font_family='sans-serif')

    ax = plt.gca()

    miny = np.min(np.asarray(list(pos.values())), axis=0)[1]
    maxy = np.max(np.asarray(list(pos.values())), axis=0)[1]
    h = maxy - miny

    for e in process_tree_graph.edges:
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
                                   color='b' if 'shuffle' in process_tree_graph.edges.get(e).keys() else "0.5",
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
                                   color='b' if 'shuffle' in process_tree_graph.edges.get(e).keys() else "0.5",
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

        label = process_tree_graph.edges.get(e)['label']
        cost = process_tree_graph.edges.get(e)['cost']
        capacity = process_tree_graph.edges.get(e)['capacity']

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
