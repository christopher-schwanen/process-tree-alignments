from pm4py.objects.process_tree.utils.generic import parse as parse_process_tree
import random
import utils
from pm4py.objects.log.obj import EventLog
from datetime import datetime
from pathlib import Path

import numpy as np
import pm4py

random.seed(42)

def boring_palindrom(m,n, number_of_traces=5):
    assert m > 1
    assert n > 1
    a = "'a'"
    b = "'b'"
    # Process Tree
    w = f"->({(a + ',')*(m-1) + a + ',' + b + ',' + (a + ',')*(m-1) + a})"
    t = f"+({(w + ',') * (n-1) + w})"
    pt = parse_process_tree(t)
    print(pt)
    # Traces
    eventlog = EventLog()
    for i in range(number_of_traces):
        w = []
        for j in range(n):
            w += ['a'] * random.randint(m, 2*m) + ['b']
        fehlende_a = (2*m*n + n - len(w))
        w += ['a'] * (fehlende_a)
        w = utils.create_trace(w, name=f"trace{i}")
        eventlog.append(w)

    return pt, eventlog


def we_love_shuffle(n, number_of_traces = 5):
    assert n > 1
    # Process Tree
    w = "->('a', 'a', 'b')"
    t = f"+({(w + ',') * (n-1) + w})"
    pt = parse_process_tree(t)

    segments = [ ['a', 'b', 'a'], ['a','a','b'], ['b', 'a', 'a']]
    # Traces
    eventlog = EventLog()
    for i in range(number_of_traces):
        w = []
        for _ in range(n):
            w += random.choice(segments)
        w = utils.create_trace(w, name=f"trace{i}")
        eventlog.append(w)

    return pt, eventlog

             
if __name__ == "__main__":
    # boring_palindrom(2, 2)
    data_path = Path("data").resolve()
    pt, eventlog = boring_palindrom(5, 5, number_of_traces=10)
    pm4py.write_ptml(pt, str(data_path / "boring_palindrom.ptml"))
    pm4py.write_xes(eventlog, str(data_path / "boring_palindrom.xes"))