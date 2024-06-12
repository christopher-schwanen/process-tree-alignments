import csv
from pathlib import Path
from timeit import Timer

import pandas as pd
from pm4py import ProcessTree, PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net.algorithm import (__get_variants_structure as get_variants,
                                                                   apply as pm4py_align_petri_net)
from pm4py.algo.conformance.alignments.process_tree.algorithm import apply as pm4py_align_process_tree
from pm4py.objects.conversion.process_tree.converter import apply as process_tree_to_petri_net
from pm4py.objects.log.obj import Trace, EventLog

from process_tree_alignment import align
from process_tree_graph import ProcessTreeGraph


def evaluate_trace(trace: Trace,
                   process_tree: ProcessTree,
                   process_tree_graph: ProcessTreeGraph,
                   accepting_petri_net: tuple[PetriNet, Marking, Marking],
                   repeat: int = 10,
                   ) -> tuple[
                                  tuple[list[float], list[float], list[float]],
                                  tuple[float, float, float],
                              ]:
    cost1, cost2, cost3 = None, None, None

    def align_wrapper() -> None:
        nonlocal cost1
        cost1 = align(trace, process_tree_graph)

    def alignments_wrapper() -> None:
        nonlocal cost2
        cost2 = pm4py_align_process_tree(trace, process_tree)['cost']

    def alignments_petri_net_wrapper() -> None:
        nonlocal cost3
        cost3 = pm4py_align_petri_net(trace, *accepting_petri_net)['cost']

    times1 = Timer(align_wrapper).repeat(repeat=repeat, number=1)
    times2 = Timer(alignments_wrapper).repeat(repeat=repeat, number=1)
    times3 = Timer(alignments_petri_net_wrapper).repeat(repeat=repeat, number=1)

    return (times1, times2, times3), (cost1, cost2, cost3)


def evaluate_event_log(event_log: EventLog | pd.DataFrame,
                       process_tree: ProcessTree,
                       repeat: int = 10,
                       result_path: str | Path = "output",
                       file_tag: str = "",
                       ) -> None:
    if isinstance(result_path, str):
        result_path = Path(result_path)
    if not result_path.is_dir():
        result_path.mkdir()

    # Build process tree graph
    process_tree_graph = ProcessTreeGraph(process_tree)
    # Build Petri net
    accepting_petri_net = process_tree_to_petri_net(process_tree)

    with (open(result_path / f"times{file_tag}.csv", "w", newline='') as f,
          open(result_path / f"costs{file_tag}.csv", "w", newline='') as g):
        time_writer = csv.writer(f, delimiter=",")
        cost_writer = csv.writer(g, delimiter=",")
        # Get trace variant results
        for variant, trace in zip(*get_variants(event_log, None)):
            results_time, results_cost = evaluate_trace(trace, process_tree, process_tree_graph, accepting_petri_net, repeat)
            for r in zip(*results_time):
                time_writer.writerow([*r, variant])
            cost_writer.writerow([*results_cost, variant])

    return
