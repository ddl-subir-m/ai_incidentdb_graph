import json

from data_handling.data_loaders import load_json_data, sample_incidents
from graph_operations.graph_builder import build_graph, prune_nodes_by_degree_threshold
from graph_operations.graph_utils import save_nodes_to_csv, visualize_graph, get_top_percentile_scores, \
    check_goals_and_run_pagerank, display_pagerank_scores
from llm_chain.node_name_extraction import extract_goal_tech

file_path = 'data/aiidb_full.json'
incident_data = load_json_data(file_path)
# incident_data = sample_incidents(incident_data, n=30)
G = build_graph(incident_data)
G = prune_nodes_by_degree_threshold(G, degree_threshold=1)


def failure_list(description: str, top_percentile: int = 85):
    extraction_result = extract_goal_tech(description)
    if "error" not in extraction_result:
        specified_goals = extraction_result.get("goals", [])
        specified_technologies = extraction_result.get("technologies", [])
        start_nodes = set(specified_goals).union(set(specified_technologies))
        result = check_goals_and_run_pagerank(G, start_nodes)
        if isinstance(result, str):
            return json.dumps({"status": "info", "message": result})
        else:
            top_scores = get_top_percentile_scores(result, percentile=top_percentile)
            if top_scores:  # Ensure there is data to process
                # Extract names and scores into separate lists
                failures = list(top_scores.keys())
                scores = list(top_scores.values())

                # Construct and return the JSON object containing these lists
                return json.dumps({
                    "status": "success",
                    "failure_modes": failures,
                    "scores": scores
                })
            else:
                # Return an error JSON if no data is available
                return json.dumps({
                    "status": "error",
                    "message": f"No data available for the top {top_percentile} percentile"
                })
    else:
        return json.dumps({"status": "info", "message": "No goals or technologies extracted."})


if __name__ == "__main__":
    print(failure_list("We have developed a chatbot that uses a transformer model for question answering."))
