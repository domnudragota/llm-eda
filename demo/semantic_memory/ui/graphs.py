import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

RELATION_COLORS = {
    'synonym': '#3b82a0',
    'antonym': '#be5266',
    'hypernym': '#8b69b3',
    'hyponym': '#cf942c',
    'definition token': '#398270'
}


def association_subgraph(graph, cue, limit):
    neighbors = sorted(graph[cue], key=lambda n: (-graph[cue][n].get('weight', 1), str(n)))
    shown = graph.subgraph([cue, *neighbors[:limit - 1]]).copy()

    shown.remove_edges_from([(u, v) for u, v in shown.edges() if cue not in (u, v)])
    return shown


def reference_subgraph(graph, cue, limit):
    buckets = {}
    for neighbor in sorted(graph[cue]):
        relation = graph[cue][neighbor]['evidence'][0]['relation']
        buckets.setdefault(relation, []).append(neighbor)

    selected = []
    while any(buckets.values()) and len(selected) < limit - 1:
        for relation in sorted(buckets):
            if buckets[relation] and len(selected) < limit - 1:
                selected.append(buckets[relation].pop(0))

    return graph.subgraph([cue, *selected]).copy()


def graph_figure(graph, cue, reference=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    positions = nx.spring_layout(graph, seed=7, weight=None)
    edges = list(graph.edges())
    colors = [RELATION_COLORS[graph[u][v]['evidence'][0]['relation']]
              if reference else '#96a7b3' for u, v in edges]

    nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=edges, edge_color=colors, alpha=.55)
    nx.draw_networkx_nodes(graph, positions, ax=ax,
                           node_color=['#da7545' if n == cue else '#cde5eb' for n in graph],
                           node_size=[650 if n == cue else 170 for n in graph])
    nx.draw_networkx_labels(graph, positions, ax=ax,
                            labels={n: textwrap.fill(n, 16) for n in graph}, font_size=10)

    ax.margins(.22)
    ax.axis('off')
    fig.tight_layout()

    return fig


def activation_subgraph(graph, prime, targets, values, limit):
    selected = set([prime, *targets])

    if len(selected) > limit:
        raise ValueError('Node limit must include the prime and all selected targets')
    if not selected <= set(graph) or not selected <= set(values.index):
        raise ValueError('Prime and targets must occur in both the original graph and matrix')

    for target in targets:
        try:
            path = nx.shortest_path(graph, prime, target)
        except nx.NetworkXNoPath:
            continue
        if len(selected.union(path)) <= limit:
            selected.update(path)

    neighbors = sorted(graph[prime], key=lambda n: (-graph[prime][n].get('weight', 1), str(n)))
    for neighbor in neighbors[:12]:
        if len(selected) < limit and neighbor in values.index:
            selected.add(neighbor)

    return graph.subgraph(sorted(selected)).copy()


def activation_graph_figure(graph, prime, targets, values):
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = nx.spring_layout(graph, seed=7, weight=None)
    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=.45,
                           edge_color='#8596a3', arrows=False, node_size=700)

    others = [node for node in graph if node != prime]
    color_scale = plt.Normalize(0, 1)
    if others:
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=others,
                               node_size=[700 if node in targets else 220 for node in others],
                               node_color=[float(values.loc[node]) for node in others],
                               cmap='viridis', vmin=0, vmax=1, edgecolors='#71808c')

    nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[prime], node_shape='D',
                           node_size=1000, node_color=[float(values.loc[prime])],
                           cmap='viridis', vmin=0, vmax=1, edgecolors='#e77a32', linewidths=3)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=10,
                            labels={node: textwrap.fill(node, 18) for node in graph})

    fig.colorbar(plt.cm.ScalarMappable(norm=color_scale, cmap='viridis'), ax=ax,
                 label='Original normalized final activation (given the selected prime)')

    ax.margins(.2)
    ax.axis('off')
    fig.tight_layout()

    return fig
