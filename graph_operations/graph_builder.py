import networkx as nx

def build_graph(incident_data):
    """Construct a directed graph from the incident data."""
    G = nx.DiGraph()
    processed_pairs = {}
    for incident in incident_data:
        incident_id = incident.get('incident_id')
        goals, technologies, failures = set(), set(), set()
        for classification in incident.get('classifications', []):
            parts = classification.split(':')
            category = ':'.join(parts[:-1]).strip()
            entity = parts[-1].strip()
            if category.endswith("Goal"):
                goals.add(entity)
                G.add_node(entity, type='Goal')
            elif category.endswith("Technology"):
                technologies.add(entity)
                G.add_node(entity, type='Technology')
            elif category.endswith("Technical Failure"):
                failures.add(entity)
                G.add_node(entity, type='Failure')
        create_edges(G, goals, technologies, failures, incident_id, processed_pairs)
    return G

def create_edges(G, goals, technologies, failures, incident_id, processed_pairs):
    """Helper function to create edges in the graph."""
    for goal in goals:
        for tech in technologies:
            add_edge(G, goal, tech, 'Goal-Tech', incident_id, processed_pairs)
        for failure in failures:
            add_edge(G, goal, failure, 'Goal-Failure', incident_id, processed_pairs)
    for tech in technologies:
        for failure in failures:
            add_edge(G, tech, failure, 'Tech-Failure', incident_id, processed_pairs)

def add_edge(G, src, dst, edge_type, incident_id, processed_pairs):
    """Add an edge to the graph if not already processed."""
    edge_key = (src, dst, edge_type, incident_id)
    if edge_key not in processed_pairs:
        weight = G.get_edge_data(src, dst, {'weight': 0})['weight'] + 1
        G.add_edge(src, dst, weight=weight)
        processed_pairs[edge_key] = True

def prune_nodes_by_degree_threshold(G, degree_threshold=0):
    """Remove nodes with a degree less than the specified threshold."""
    underconnected_nodes = [node for node, degree in G.degree() if degree <= degree_threshold]
    G.remove_nodes_from(underconnected_nodes)
    return G
