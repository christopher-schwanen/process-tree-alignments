import csv
import random
from multiprocessing import Process, Queue, Value, cpu_count
from pathlib import Path
from timeit import Timer
from typing import Optional

import numpy as np
import pandas as pd
from pm4py import ProcessTree, PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net.algorithm import (__close_progress_bar as close_progress_bar,
                                                                   __get_progress_bar as get_progress_bar,
                                                                   __get_variants_structure as get_variants,
                                                                   apply as pm4py_align_petri_net)
from pm4py.algo.conformance.alignments.process_tree.algorithm import apply as pm4py_align_process_tree
from pm4py.objects.conversion.process_tree.converter import apply as process_tree_to_petri_net
from pm4py.objects.log.obj import Trace, EventLog

from process_tree_alignment import align
from process_tree_graph import ProcessTreeGraph


# Set seed for reproducibility
random.seed(42)
# Number of CPUs to use
N_CPUS = cpu_count()
N_WORKERS = N_CPUS // 2 - 2 if N_CPUS < 32 else N_CPUS - 16
TIMEOUT = 65
MAX_TRACE_VARIANTS = 1000
OFFSET = 0


def align_wrapper(cost1: Value, trace: Trace, process_tree_graph: ProcessTreeGraph) -> None:
    cost1.value = align(trace, process_tree_graph)


def alignments_wrapper(cost2: Value, trace: Trace, process_tree: ProcessTree) -> None:
    cost2.value = pm4py_align_process_tree(trace, process_tree)['cost']


def alignments_petri_net_wrapper(cost3: Value,
                                 trace: Trace,
                                 accepting_petri_net: tuple[PetriNet, Marking, Marking]
                                 ) -> None:
    cost3.value = pm4py_align_petri_net(trace, *accepting_petri_net)['cost']


def evaluate_trace(trace: Trace,
                   process_tree: ProcessTree,
                   process_tree_graph: ProcessTreeGraph,
                   accepting_petri_net: tuple[PetriNet, Marking, Marking],
                   repeat: int = 10,
                   ) -> tuple[
                                  tuple[list[float], list[float], list[float]],
                                  tuple[float, float, float],
                              ]:
    
    cost1, cost2, cost3 = Value('d', -1.0), Value('d', -1.0), Value('d', -1.0)

    def process_with_timeout(process: Process) -> None:
        process.start()
        process.join(TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError
        process.close()

    def align_with_timeout() -> None:
        process_with_timeout(Process(target=align_wrapper, args=(cost1, trace, process_tree_graph)))

    def alignments_with_timeout() -> None:
        process_with_timeout(Process(target=alignments_wrapper, args=(cost2, trace, process_tree)))

    def alignments_petri_net_with_timeout() -> None:
        process_with_timeout(Process(target=alignments_petri_net_wrapper, args=(cost3, trace, accepting_petri_net)))

    try:
        times1 = Timer(align_with_timeout).repeat(repeat=repeat, number=1)
    except TimeoutError:
        times1 = [TIMEOUT] * repeat
    try:
        times2 = Timer(alignments_with_timeout).repeat(repeat=repeat, number=1)
    except TimeoutError:
        times2 = [TIMEOUT] * repeat
    try:
        times3 = Timer(alignments_petri_net_with_timeout).repeat(repeat=repeat, number=1)
    except TimeoutError:
        times3 = [TIMEOUT] * repeat

    if min(times1) >= TIMEOUT:
        cost1.value = -1.0
    if min(times2) >= TIMEOUT:
        cost2.value = -1.0
    if min(times3) >= TIMEOUT:
        cost3.value = -1.0

    return (times1, times2, times3), (cost1.value, cost2.value, cost3.value)


# Define a process which will be in charge of storing the results of the benchmark
def store_results(q: Queue,
                  result_path: Path,
                  file_tag: str = "",
                  number_of_variants: Optional[int] = None,
                  ) -> None:
    with (open(result_path / f"times{file_tag}.csv", "w", newline='') as file_time,
          open(result_path / f"costs{file_tag}.csv", "w", newline='') as file_cost):
        time_writer = csv.writer(file_time, delimiter=",")
        cost_writer = csv.writer(file_cost, delimiter=",")

        # PM4Py progress bar
        progress = get_progress_bar(number_of_variants, None) if number_of_variants else None

        while True:
            data = q.get()
            if data == 'Ende':
                if progress:
                    close_progress_bar(progress)
                break
            for time in data[0]:
                time_writer.writerow(time)
            cost_writer.writerow(data[1])
            file_time.flush()
            file_cost.flush()
            if progress:
                progress.update()


# Define the worker processes which run the benchmarks for the different traces
# Each worker gets a chunk as input and outputs results into the shared queue
def worker_chunk(q: Queue,
                 chunk: list,
                 process_tree: ProcessTree,
                 process_tree_graph: ProcessTreeGraph,
                 accepting_petri_net: tuple[PetriNet, Marking, Marking],
                 repeat: int,
                 ) -> None:
    for variant, trace in chunk:
        results_time, results_cost = evaluate_trace(trace, process_tree, process_tree_graph, accepting_petri_net, repeat)
        time_results = []
        for r in zip(*results_time):
            time_results.append([*r, variant])
        cost_results = [*results_cost, variant]
        results = (time_results, cost_results)
        q.put(results)
    return


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

    # Get trace variant results and shuffle them
    trace_variants = list(zip(*get_variants(event_log, None)))
    random.shuffle(trace_variants)
    # Check if number of trace_variants is below max_trace_variants
    if len(trace_variants) > max_trace_variants:
        trace_variants = trace_variants[OFFSET:(max_trace_variants+OFFSET)]

    # Collect the work to do:
    trace_variants = list(trace_variants)
    if (n := len(trace_variants)) == 0:
        print("No trace variants found.")
        return
    print(f"Number of trace variants: {n}")

    # We create a shared queue to store the results of the benchmark
    q = Queue()

    # Start writer process to store the results
    writer = Process(target=store_results, args=(q, result_path, file_tag, n))
    writer.start()

    # Split the trace variants into chunks of size N_WORKERS
    # compute chunk sizes so that N_WORKERS are used
    chunk_size = int(np.ceil(n / N_WORKERS))
    trace_variants_chunks = [trace_variants[i:i + chunk_size] for i in range(0, n, chunk_size)]

    # For each chunk, we start a worker process
    workers = []
    for chunk in trace_variants_chunks:
        p = Process(target=worker_chunk, args=(q, chunk, process_tree, process_tree_graph, accepting_petri_net, repeat))
        p.start()
        workers.append(p)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Send signal to writer to stop
    q.put('Ende')
    writer.join()

    return
