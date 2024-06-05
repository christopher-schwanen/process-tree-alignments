import gurobipy as gp
from gurobipy import GRB
from pm4py.objects.log.obj import Trace

from process_tree_graph import ProcessTreeGraph


def align(trace: Trace, process_tree_graph: ProcessTreeGraph) -> float:
    m = gp.Model("process_tree_alignment")

    # Flow variables including arc capacity constraints and objective function (minimize is default)
    x = m.addVars(len(trace) + 1, process_tree_graph.edges,
                  lb=0, ub={(i, *e): c for i in range(len(trace) + 1)
                            for *e, c in process_tree_graph.edges(keys=True, data='capacity')},
                  obj={(i, *e): c for i in range(len(trace) + 1)
                       for *e, c in process_tree_graph.edges(keys=True, data='cost')},
                  vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(range(1, len(trace) + 1), process_tree_graph.nodes,
                  lb=0, ub=1,
                  obj=1,
                  vtype=GRB.CONTINUOUS, name="y")
    sync_edges = {i+1: [e for e in process_tree_graph.edges
                        if process_tree_graph.edges.get(e).get('label') == a.get('concept:name')]
                  for i, a in enumerate(trace)}
    z = m.addVars([(i, *e) for i in sync_edges for e in sync_edges[i]],
                  lb=0, ub={(i, *e): process_tree_graph.edges.get(e).get('capacity') for i in sync_edges
                            for e in sync_edges[i]},
                  obj={(i, *e): 1 - cost if (cost := process_tree_graph.edges.get(e).get('cost')) > 1 else 0
                       for i in sync_edges for e in sync_edges[i]},
                  vtype=GRB.CONTINUOUS, name="z")

    # Flow conservation constraints
    if len(trace) == 0:
        m.addConstrs(x.sum(0, '*', v, '*') - x.sum(0, v, '*', '*')
                     == (0 if process_tree_graph.nodes.get(v).get('source')
                              == process_tree_graph.nodes.get(v).get('sink')
                         else -1 if process_tree_graph.nodes.get(v).get('source') else 1)
                     for v in process_tree_graph.nodes)
    else:
        m.addConstrs(x.sum(i, '*', v, '*')
                     + (y[i, v] + z.sum(i, '*', v, '*') if i > 0 else 0)
                     - x.sum(i, v, '*', '*')
                     - (y[i + 1, v] + z.sum(i + 1, v, '*', '*') if i < len(trace) else 0)
                     == (-1 if i == 0 and process_tree_graph.nodes.get(v).get('source')
                         else 1 if i == len(trace) and process_tree_graph.nodes.get(v).get('sink') else 0)
                     for i in range(len(trace) + 1) for v in process_tree_graph.nodes)

    # Synchronization variables and constraints
    shuffles = {(k, i): w for k, v in process_tree_graph.nodes._nodes.items() if v.get('shuffle') for i, w in
                enumerate(v['shuffle'])}
    s = m.addVars(len(trace) + 1, shuffles.keys(), vtype=GRB.BINARY, name="s")
    m.addConstrs(gp.quicksum(x[(i, *e)] for e in shuffles[(v, j)])
                 == s[(i, v, j)] / process_tree_graph.nodes.get(v)['iac']
                 for v, j in shuffles
                 for i in range(len(trace)+1))

    # Duplicate labels constraints
    m.addConstrs(gp.quicksum(z[(i, *e)] * process_tree_graph.edges.get(e).get('cost') for e in sync_edges[i]) <= 1
                 for i in sync_edges if len(sync_edges[i]) > 1)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        return m.objVal
    raise Exception(f"Optimization failed with status {m.status}")
