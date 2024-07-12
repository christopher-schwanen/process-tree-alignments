from typing import Optional

import pandas as pd
from pm4py import ProcessTree, discover_process_tree_inductive
from pm4py.objects.log.obj import Trace, Event, EventLog


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
