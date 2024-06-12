from datetime import datetime
from pathlib import Path

import numpy as np
import pm4py

from evaluation import evaluate_event_log
from process_tree_alignment import align
from process_tree_graph import ProcessTreeGraph
from utils import create_trace, discover_process_tree


def example() -> None:
    from pm4py.algo.conformance.alignments.process_tree.algorithm import apply as alignments
    from pm4py.objects.process_tree.utils.generic import parse as parse_process_tree
    process_tree = parse_process_tree("->(*(X(->('a', 'b'), +('c', 'd')), tau), +('e', 'a'))")
    pm4py.view_process_tree(process_tree)
    graph = ProcessTreeGraph(process_tree)
    graph.view(rad=0.1, rad_offset=0.2)

    trace = create_trace(["a", "b", "c", "f"], name="trace")

    print(f"The alignment cost for trace {[v['concept:name'] for i, v in enumerate(trace)]} are: "
          f"{(a1 := align(trace, graph))} \t{(a2 := alignments(trace, process_tree)['cost'])} \t({np.isclose(a1, a2)})")


if __name__ == "__main__":
    data_path = Path("data").resolve()
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir()
    for xes_file in data_path.glob("*.xes"):
        if not xes_file.is_file():
            continue
        cur_path = result_path / xes_file.stem
        cur_path.mkdir()
        event_log = pm4py.read_xes(str(xes_file))
        for noise_threshold, noise_tag in [(0.0, "00"), (0.1, "10"), (0.25, "25"), (0.5, "50")]:
            process_tree = discover_process_tree(event_log, noise_threshold=noise_threshold)
            pm4py.write_ptml(process_tree, str(cur_path / f"{xes_file.stem}_pt{noise_tag}.ptml"))
            evaluate_event_log(event_log, process_tree, repeat=10, result_path=cur_path, file_tag=f"_pt{noise_tag}")
    # example()
