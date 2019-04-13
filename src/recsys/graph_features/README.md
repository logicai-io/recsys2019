## Graph Features Description

Graph is based on impressions.

### Pairs of impressions:

- edges_records_df: count of neighbors and neighbors statistics for pairs of impressions items, list of nodes connected to each node, [ref](https://networkx.github.io/documentation/stable/reference/generated/networkx.classes.function.neighbors.html?highlight=neighbors#networkx.classes.function.neighbors)
- graph_matched_dict: whether a pair of items represents a valid matching (no two distinct edges share a common endpoint), [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.is_matching.html?highlight=is_matching#networkx.algorithms.matching.is_matching)

Above can be merged onto each pair of impressions.

### Features per impressions item:

- cluster_dict: clustering coefficient for each item, [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html)
- cluster_triangles_dict: number of triangles for each item, [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.triangles.html#networkx.algorithms.cluster.triangles)
- avg_neighbor_deg_dict: average degree of neighborhood of each node, [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_neighbor_degree.html?highlight=average_neighbor_degree#networkx.algorithms.assortativity.average_neighbor_degree)
- pagerank_dict: pagerank measure for each item, [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?highlight=pagerank#networkx.algorithms.link_analysis.pagerank_alg.pagerank)

Above can be merged based on each impressions item ID.

### Other:

- avg_deg_connect_dict: average degree connectivity of graph. describes connectivity for each number of neighbors. can be merged based on `node*_neighbor_count`  to `edges_records_df`, [ref](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_degree_connectivity.html?highlight=average_degree_connectivity#networkx.algorithms.assortativity.average_degree_connectivity)
