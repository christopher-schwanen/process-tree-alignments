from collections import Counter

import gurobipy as gp
import networkx as nx
from gurobipy import GRB
from pm4py.objects.log.obj import Trace


def align(trace: Trace, process_tree_graph: nx.MultiDiGraph) -> float:
    m = gp.Model("process_tree_alignment")
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
    sync_edges = [(i + 1, *e) for e in process_tree_graph.edges for i, a in enumerate(trace)
                  if process_tree_graph.edges.get(e).get('label') == a.get('concept:name')]
    z = m.addVars(list(sync_edges),
                  lb=0, ub={(i, *e): process_tree_graph.edges.get(e).get('capacity') for i, *e in sync_edges},
                  obj={(i, *e): 1 - cost if (cost := process_tree_graph.edges.get(e).get('cost')) > 1 else 0
                       for i, *e in sync_edges},
                  vtype=GRB.CONTINUOUS, name="z")

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

    shuffles = {(v, j): {'iac': process_tree_graph.nodes.get(v).get('shuffle')[j],
                         'edges': [e for e in process_tree_graph.in_edges(v, keys=True)
                                   if process_tree_graph.edges.get(e).get('shuffle') == j]}
                for v in process_tree_graph.nodes if process_tree_graph.nodes.get(v).get('shuffle')
                for j in range(len(process_tree_graph.nodes.get(v).get('shuffle')))}
    s = m.addVars(len(trace) + 1, shuffles.keys(), vtype=GRB.BINARY, name="s")
    m.addConstrs(gp.quicksum(x[(i, *e)] for e in shuffles[(v, j)]['edges'])
                 == s[(i, v, j)] / shuffles[(v, j)]['iac']
                 for v, j in shuffles
                 for i in range(len(trace)+1))

    i_par_sync = [k for k, v in Counter(i for i, *e in sync_edges).items() if v > 1]
    sync_edges_iacs = {(i, *e): process_tree_graph.edges.get(e).get('cost') for i, *e in sync_edges if i in i_par_sync}
    m.addConstrs(z.prod(sync_edges_iacs, i, '*', '*', '*') <= 1 for i in i_par_sync)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        return m.objVal
    raise Exception(f"Optimization failed with status {m.status}")
