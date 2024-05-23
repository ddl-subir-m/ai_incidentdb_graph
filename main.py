from data_handling.data_loaders import load_json_data, sample_incidents
from graph_operations.graph_builder import build_graph, prune_nodes_by_degree_threshold
from graph_operations.graph_utils import save_nodes_to_csv, visualize_graph, get_top_percentile_scores, \
    check_goals_and_run_pagerank, display_pagerank_scores
from llm_chain.node_name_extraction import extract_goal_tech


def main():
    # file_path = 'data/aiidb_full.json'
    file_path = '/mnt/code/data/aiidb_full.json'
    incident_data = load_json_data(file_path)
    # incident_data = sample_incidents(incident_data, n=30)
    G = build_graph(incident_data)
    G = prune_nodes_by_degree_threshold(G, degree_threshold=1)
    # save_nodes_to_csv(G, filename="goals_technologies.csv")
    # visualize_graph(G)
    answer_text = "We have developed a chatbot that uses a transformer model for question answering."
    extraction_result = extract_goal_tech(answer_text)
    if "error" not in extraction_result:
        specified_goals = extraction_result.get("goals", [])
        specified_technologies = extraction_result.get("technologies", [])
        start_nodes = set(specified_goals).union(set(specified_technologies))
        result = check_goals_and_run_pagerank(G, start_nodes)
        if isinstance(result, str):
            print(result)  # Handling the string message for no results
        else:
            display_pagerank_scores(get_top_percentile_scores(result, percentile=85), G)
    else:
        print(extraction_result["error"])


if __name__ == "__main__":
    main()
