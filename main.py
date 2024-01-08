import numpy as np
import pm4py

import utils
from process_tree_alignment import align
from process_tree_graph import build_process_tree_graph, view_process_tree_graph


def example() -> None:
    from pm4py.algo.conformance.alignments.process_tree.algorithm import apply as alignments
    from pm4py.objects.process_tree.utils.generic import parse as parse_process_tree
    process_tree = parse_process_tree("->(*(X(->('a','b'), +('c','d')), tau),+('e', 'a'))")
    pm4py.view_process_tree(process_tree)
    graph = build_process_tree_graph(process_tree)
    view_process_tree_graph(graph, rad=0.1, rad_offset=0.2)

    trace = utils.create_trace(["a", "b", "c", "f"], name="trace")

    print(f"The alignment cost for trace {[v['concept:name'] for i, v in enumerate(trace)]} are: "
          f"{(a1 := align(trace, graph))} \t{(a2 := alignments(trace, process_tree)['cost'])} \t({np.isclose(a1, a2)})")


if __name__ == "__main__":
    example()
