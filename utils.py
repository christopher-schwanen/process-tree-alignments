from typing import Optional

from pm4py.objects.log.obj import Trace, Event


def create_trace(labels: list[str], name: Optional[str] = None) -> Trace:
    trace = Trace()
    if name is not None:
        trace.attributes["concept:name"] = name
    for label in labels:
        event = Event()
        event["concept:name"] = label
        trace.append(event)
    return trace
