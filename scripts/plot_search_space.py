import sys
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read_rules(file_path):
    rules = []
    scores = {}
    raw_scores = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            antecedent = tuple(sorted(map(int, row['antecedent'].strip('{}').split(';'))))
            consequent = tuple(sorted(map(int, row['consequent'].strip('{}').split(';'))))
            rule = (antecedent, consequent)
            score = int(row['freqX']) + int(row['freqY']) + int(row['freqZ'])
            rules.append(rule)
            raw_scores.append(score)
    
    # Normalization of scores
    max_score = max(raw_scores)
    min_score = min(raw_scores)
    for rule, score in zip(rules, raw_scores):
        normalized_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 0
        scores[rule] = normalized_score

    return rules, scores

def out_ranking_certainty(scoreA, scoreB):
    csi = 10
    # return (scoreA - scoreB) / 2 + 0.5
    return np.exp(csi*scoreA) / (np.exp(csi*scoreA) + np.exp(csi*scoreB)) 

def create_graph(rules, scores):
    G = nx.Graph()
    for rule in rules:
        G.add_node(rule, score=scores[rule], size=300)  # Fixed size for all nodes
    
    for i, rule1 in enumerate(rules):
        neighbors = []
        for j, rule2 in enumerate(rules):
            if i != j and are_neighbors(rule1, rule2):
                neighbors.append(rule2)
                
        for rule2 in neighbors:
            weight = out_ranking_certainty(scores[rule2], scores[rule1]) / len(neighbors)
            G.add_edge(rule1, rule2, weight=weight)
        
        # Calculate self-loop weight based on neighbors using the corrected formula
        if neighbors:
            self_weight = sum(out_ranking_certainty(scores[rule1], scores[rule2]) for rule2 in neighbors) / len(neighbors)
            G.add_edge(rule1, rule1, weight=self_weight)

    return G

def compute_statistics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    if num_nodes > 1:
        connectivity_ratio = num_edges / (num_nodes * (num_nodes - 1) / 2)
    else:
        connectivity_ratio = 0

    # Retrieve edge weights from the graph
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    avg_edge_weight = np.mean(edge_weights) if edge_weights else 0
    max_edge_weight = max(edge_weights) if edge_weights else 0
    min_edge_weight = min(edge_weights) if edge_weights else 0

    neighbor_counts = [len(list(nx.neighbors(G, node))) for node in G.nodes()]
    avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
    max_neighbors = max(neighbor_counts) if neighbor_counts else 0
    min_neighbors = min(neighbor_counts) if neighbor_counts else 0

    score_diffs = []
    for node in G.nodes():
        node_score = G.nodes[node]['score']
        neighbors = nx.neighbors(G, node)
        score_diffs.extend(abs(node_score - G.nodes[neigh]['score']) for neigh in neighbors)

    avg_score_diff = np.mean(score_diffs) if score_diffs else 0
    max_score_diff = max(score_diffs) if score_diffs else 0
    min_score_diff = min(score_diffs) if score_diffs else 0

    print(f"Total number of nodes: {num_nodes}")
    print(f"Total number of edges: {num_edges}")
    print(f"Average number of neighbors: {avg_neighbors:.2f}")
    print(f"Maximum number of neighbors: {max_neighbors}")
    print(f"Minimum number of neighbors: {min_neighbors}")
    print(f"Average difference between node's scores and its neighbors: {avg_score_diff:.2f}")
    print(f"Maximum difference between node's scores and its neighbors: {max_score_diff:.2f}")
    print(f"Average edge weight: {avg_edge_weight:.4f}")
    print(f"Maximum edge weight: {max_edge_weight:.4f}")
    print(f"Minimum edge weight: {min_edge_weight:.4f}")
    print(f"Connectivity ratio: {connectivity_ratio:.4f}")

def are_neighbors(rule1, rule2):
    antecedent1, _ = rule1
    antecedent2, _ = rule2
    return len(set(antecedent1).symmetric_difference(set(antecedent2))) == 1

def plot_graph(G):
    pos = nx.spring_layout(G)

    # Prepare edge colors with varying alpha values based on edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    min_weight, max_weight = min(edge_weights.values()), max(edge_weights.values())
    edge_alphas = [(edge_weights[(u, v)] - min_weight) / (max_weight - min_weight) for u, v in G.edges()]  # Normalize weights for alpha

    node_scores = nx.get_node_attributes(G, 'score')
    min_score, max_score = min(node_scores.values()), max(node_scores.values())
    node_colors = [plt.cm.RdBu((score - min_score) / (max_score - min_score)) for score in node_scores.values()]  # Color nodes from blue to red based on scores

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    # Draw edges with alpha proportional to normalized weight
    for (u, v), alpha in zip(G.edges(), edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color='black', alpha=alpha)

    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    rules, scores = read_rules(file_path)
    G = create_graph(rules, scores)
    compute_statistics(G)
    plot_graph(G)

if __name__ == "__main__":
    main()
