def build_association_layer(cleaned_responses, original):
    edges = original.FA_edgeList(cleaned_responses.copy())
    directed = original.graphFromEdgeList(edges, directed=True, weighted=True)
    undirected = original.makeUndirected(directed)
    lexical = original.WNfilter(undirected)
    repeated = original.idiosynfilter(lexical)
    return original.CC(repeated)
