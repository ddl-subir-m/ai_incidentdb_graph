import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def save_nodes_to_csv(G, filename="goals_technologies.csv"):
    """Save unique node types to CSV."""
    goals = list({data.get('name', n) for n, data in G.nodes(data=True) if data.get('type') == 'Goal'})
    technologies = list({data.get('name', n) for n, data in G.nodes(data=True) if data.get('type') == 'Technology'})
    failures = list({data.get('name', n) for n, data in G.nodes(data=True) if data.get('type') == 'Failure'})
    max_length = max(len(goals), len(technologies), len(failures))
    goals.extend([None] * (max_length - len(goals)))
    technologies.extend([None] * (max_length - len(technologies)))
    failures.extend([None] * (max_length - len(failures)))
    df = pd.DataFrame({'Goals': goals, 'Technologies': technologies, 'Failures': failures})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def visualize_graph(G):
    """Visualize the graph using a shell layout with nodes colored by their type."""
    shell_order = ['Failure', 'Technology', 'Goal']
    shells = {typ: [] for typ in shell_order}
    for node, data in G.nodes(data=True):
        if data.get('type') in shells:
            shells[data.get('type')].append(node)
    shell_positions = nx.shell_layout(G, [shells[typ] for typ in shell_order])
    node_colors = {'Goal': 'lightblue', 'Technology': 'lightgreen', 'Failure': 'salmon'}
    colors = [node_colors[G.nodes[node]['type']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, shell_positions, node_color=colors, node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, shell_positions, width=2)
    nx.draw_networkx_labels(G, shell_positions, font_size=10, font_family='sans-serif')
    plt.title('AI Incident Graph with Colored Nodes Based on Type')
    plt.axis('off')
    plt.show()


def personalized_pagerank(G, start_nodes, alpha=0.85):
    """Calculate personalized PageRank based on specific start nodes."""
    personalization = {node: 0.01 for node in G.nodes()}  # Small base value for all nodes
    for node in start_nodes:
        personalization[node] = 1.0
    total_personalization = sum(personalization.values())
    personalization = {k: v / total_personalization for k, v in personalization.items()}
    pagerank_scores = nx.pagerank(G, personalization=personalization, weight='weight', alpha=alpha)
    failure_pagerank_scores = {node: score for node, score in pagerank_scores.items() if
                               G.nodes[node].get('type') == 'Failure'}

    return failure_pagerank_scores



def get_top_percentile_scores(pagerank_scores, percentile=90):
    """Return nodes and their PageRank scores that are in the top specified percentile."""
    scores = list(pagerank_scores.values())
    threshold = np.percentile(scores, percentile)
    top_scores = {node: score for node, score in pagerank_scores.items() if score >= threshold}
    # Sort the top scores in descending order
    sorted_top_scores = dict(sorted(top_scores.items(), key=lambda item: item[1], reverse=True))

    return sorted_top_scores


def display_pagerank_scores(pagerank_scores, G, node_type='Failure'):
    """Display PageRank scores, filtering by node type."""
    filtered_scores = {node: score for node, score in pagerank_scores.items() if G.nodes[node].get('type') == node_type}

    # Sort nodes by PageRank score in descending order
    sorted_scores = sorted(filtered_scores.items(), key=lambda item: item[1], reverse=True)

    for node, score in sorted_scores:
        print(f"{node}: {score:.4f}")


def check_goals_and_run_pagerank(G, specified_goals):
    """Check if any specified goals are in the graph and run PageRank if present."""
    # Check if any specified goals are in the graph
    graph_nodes = set(G.nodes())
    valid_goals = graph_nodes.intersection(set(specified_goals))
    if not valid_goals:
        return "No search results for the specified goals."

    # Run PageRank with the valid goals
    pagerank_scores = personalized_pagerank(G, valid_goals)
    if pagerank_scores is None:
        return "No valid start nodes found in the graph."

    # Return the PageRank scores or any other analysis results
    return pagerank_scores
