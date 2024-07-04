from pm4py.objects.process_tree.utils.generic import parse as parse_process_tree
import random
import utils
import pandas as pd
from pm4py.objects.log.obj import EventLog
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pm4py
from utils import create_trace, discover_process_tree
from pm4py.algo.conformance.alignments.petri_net.algorithm import __get_variants_structure as get_variants

random.seed(42)

relevant_columns = ["case:concept:name", "concept:name", "time:timestamp"]
MAX_DISTINCTION = 3
DATA_PATH = Path("data").resolve()

def unifyTrace(trace):
    counter = {}
    counter = {a : 0 for a in trace}
    new_trace = []
    for a in trace:
        counter[a] = counter[a] + 1 if counter[a] < MAX_DISTINCTION else counter[a]
        if counter[a] > 1:
            new_trace.append(a + '__COUNT__' + str(counter[a]))
        else:
            new_trace.append(a)
    return pd.Series(new_trace, index=trace.index, name=trace.name)

def unifyXES(event_log: EventLog):
    traces = event_log.groupby("case:concept:name")['concept:name'].transform(lambda x: unifyTrace(x)).reset_index(drop=True)
    event_log['concept:name'] = traces
    return event_log

def removeCOUNTfromTree(node):
    if node.label:
        if "__COUNT__" in node.label:
            node.label = node.label.split("__COUNT__")[0]
    if node.children:
        for child in node.children:
            removeCOUNTfromTree(child)


if __name__ == "__main__":
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir()
    
    # Go through all *.xes files in the data folder and make them harder

    for xes_file in DATA_PATH.glob("*.xes"):
        # Check if the filename does not end with "_harder.xes"
        if xes_file.stem.endswith("_harder"):
            continue
        print("Processing file: ", xes_file)
        event_log = pm4py.read_xes(str(DATA_PATH / "sepsis.xes"))[relevant_columns]
        harderEventlog = unifyXES(event_log)
        pm4py.write_xes(harderEventlog, str(DATA_PATH / (xes_file.stem + "_harder.xes")))
        print("Done processing file: ", xes_file)

    # Construct Process Trees for all xes files in the data folder
    for xes_file in DATA_PATH.glob("*.xes"):
        if not xes_file.is_file():
            continue
        cur_path = result_path / xes_file.stem
        cur_path.mkdir()
        print(f"{xes_file.stem}")
        event_log = pm4py.read_xes(str(xes_file))
        # Check if in data_path there is a file with the same name as the xes file but with the extension .ptml
        if (ptml_file := DATA_PATH / f"{xes_file.stem}.ptml").is_file():
            continue
        else:
            for noise_threshold, file_tag in [(0.0, "_pt00"), (0.1, "_pt10"), (0.25, "_pt25"), (0.5, "_pt50")]:
                if (ptml_file := DATA_PATH / f"{xes_file.stem}{file_tag}.ptml").is_file():
                    continue
                else:
                    print(f"Discovering process tree with noise threshold {noise_threshold} and xes file {xes_file.stem}")
                    process_tree = discover_process_tree(event_log, noise_threshold=noise_threshold)
                    removeCOUNTfromTree(process_tree)
                    pm4py.write_ptml(process_tree, str(DATA_PATH / f"{xes_file.stem}{file_tag}.ptml"))
                print(f" -> {ptml_file.stem}")