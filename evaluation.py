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

import random
random.seed(42)

import multiprocessing

TIMEOUT = 10
MAX_TRACE_VARIANTS = 10

def evaluate_trace(trace: Trace,
                   process_tree: ProcessTree,
                   process_tree_graph: ProcessTreeGraph,
                   accepting_petri_net: tuple[PetriNet, Marking, Marking],
                   repeat: int = 10,
                   ) -> tuple[
                                  tuple[list[float], list[float], list[float]],
                                  tuple[float, float, float],
                              ]:
    costs = [None, None, None]

    def align_wrapper(costs) -> None:
        costs[0] = align(trace, process_tree_graph)


    def alignments_wrapper(costs) -> None:
        costs[1] = pm4py_align_process_tree(trace, process_tree)['cost']

    def alignments_petri_net_wrapper(costs) -> None:
        costs[2] = pm4py_align_petri_net(trace, *accepting_petri_net)['cost']


    def align_with_timeout() -> None:
        nonlocal costs
        p = multiprocessing.Process(target=align_wrapper, args=(costs,))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate()
        p.close()

    def alignments_with_timeout() -> None:
        nonlocal costs
        p = multiprocessing.Process(target=alignments_wrapper, args=(costs,))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join()
    
    def alignments_petri_net_with_timeout() -> None:
        nonlocal costs
        p = multiprocessing.Process(target=alignments_petri_net_wrapper, args=(costs,))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join()

    cost1, cost2, cost3 = costs

    times1 = Timer(align_with_timeout).repeat(repeat=repeat, number=1)
    times2 = Timer(alignments_with_timeout).repeat(repeat=repeat, number=1)
    times3 = Timer(alignments_petri_net_with_timeout).repeat(repeat=repeat, number=1)

    # if cost1 is None:
    #     cost1 = float('nan')
    #     times1 = [float('nan')]
    # if cost2 is None:
    #     cost2 = float('nan')
    #     times2 = [float('nan')]
    # if cost3 is None:
    #     cost3 = float('nan')
    #     times3 = [float('nan')]

    return (times1, times2, times3), (cost1, cost2, cost3)


def evaluate_event_log(event_log: EventLog | pd.DataFrame,
                       process_tree: ProcessTree,
                       repeat: int = 5,
                       result_path: str | Path = "output",
                       file_tag: str = "",
                       max_trace_variants: int = MAX_TRACE_VARIANTS,
                       ) -> None:
    if isinstance(result_path, str):
        result_path = Path(result_path)
    if not result_path.is_dir():
        result_path.mkdir()

    # Build process tree graph
    process_tree_graph = ProcessTreeGraph(process_tree)
    # Build Petri net
    accepting_petri_net = process_tree_to_petri_net(process_tree)

    with (open(result_path / f"times{file_tag}.csv", "w", newline='') as file_time,
          open(result_path / f"costs{file_tag}.csv", "w", newline='') as file_cost):
        time_writer = csv.writer(file_time, delimiter=",")
        cost_writer = csv.writer(file_cost, delimiter=",")
        # Get trace variant results
        trace_variants = list(zip(*get_variants(event_log, None)))
        # Check if number of trace_variants is below max_trace_variants
        if len(trace_variants) > max_trace_variants:
            trace_variants = random.sample(trace_variants, max_trace_variants)
        for variant, trace in trace_variants:
            results_time, results_cost = evaluate_trace(trace, process_tree, process_tree_graph, accepting_petri_net, repeat)
            for r in zip(*results_time):
                time_writer.writerow([*r, variant])
            cost_writer.writerow([*results_cost, variant])
            file_time.flush()
            file_cost.flush()

    return
