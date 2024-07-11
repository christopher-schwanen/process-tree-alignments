import math
from typing import Optional

import pandas as pd
from pm4py import ProcessTree, discover_process_tree_inductive
from pm4py.objects.log.obj import Trace, Event, EventLog
from pm4py.objects.process_tree.utils.generic import get_leaves


def create_trace(labels: list[str], name: Optional[str] = None) -> Trace:
    trace = Trace()
    if name is not None:
        trace.attributes["concept:name"] = name
    for label in labels:
        event = Event()
        event["concept:name"] = label
        trace.append(event)
    return trace


def discover_process_tree(event_log: EventLog | pd.DataFrame, noise_threshold: float = 0.25) -> ProcessTree:
    # Discover process tree using Inductive Miner infrequent
    return discover_process_tree_inductive(event_log, noise_threshold=noise_threshold)


def discover_process_tree_with_naive_label_splitting(event_log: pd.DataFrame,
                                                     max_distinction: int = 3,
                                                     noise_threshold: float = 0.25,
                                                     ) -> ProcessTree:
    if max_distinction < 1:
        raise ValueError("max_distinction must be at least 1.")
    digits = int(math.log10(max_distinction)) + 1

    # Add counts to activity labels
    event_log = event_log.copy()
    event_log.sort_values(['case:concept:name', 'time:timestamp', 'concept:name', 'lifecycle:transition'])
    event_log['repetition'] = event_log.groupby(['case:concept:name', 'concept:name', 'lifecycle:transition']).cumcount()
    event_log['repetition'] = event_log['repetition'] % max_distinction
    event_log['concept:name'] = event_log['concept:name'] + '_' + event_log['repetition'].astype(str).str.zfill(digits)

    process_tree = discover_process_tree_inductive(event_log, noise_threshold=noise_threshold)

    # Remove counts from process tree labels
    for leaf in get_leaves(process_tree):
        if leaf.label:
            leaf.label = leaf.label[:-(digits+1)]

    return process_tree
