import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import argparse

def load_path_results(result_file='shortest_path_result.json'):
    """Load shortest path results from JSON file"""
    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Result file {result_file} not found. Please run find_shortest_path.py first.")
        return None

def create_network_graph(csv_file, num_rows=None):
    """Create a NetworkX graph from the temporal network data"""
    # Load data
    if num_rows:
        df = pd.read_csv(csv_file, nrows=num_rows)
    else:
        df = pd.read_csv(csv_file)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with weights (travel time)
    for _, row in df.iterrows():
        duration = row['arr_time_ut'] - row['dep_time_ut']
        if duration > 0:
            G.add_edge(
                str(row['from_stop_I']), 
                str(row['to_stop_I']),
                weight=duration,
                departure=row['dep_time_ut'],
                arrival=row['arr_time_ut']
            )
    
    return G

def extract_path_vertices(path_details):
    """Extract the sequence of vertices from the path details"""
    if not path_details:
        return []
    
    vertices = []
    for node_str in path_details:
        # Parse node string to extract vertices
        # Format: "v_e{id}[{u}->{v} | dep:{t}, arr:{a}]"
        try:
            # Extract u and v from the node representation
            parts = node_str.split('[')[1].split('->')[0:2]
            u = parts[0].strip()
            v = parts[1].split(' |')[0].strip()
            
            if not vertices:  # First node
                vertices.append(u)
            vertices.append(v)
        except:
            continue
    
    return vertices

def visualize_shortest_path(result_data, csv_file, save_plot=True):
    """Create visualization of the shortest path"""
    # Create the base network graph
    G = create_network_graph(csv_file, num_rows=10000)  # Increased for better path coverage
    
    # Extract path information
    source = result_data['source']
    destination = result_data['destination']
    journey_time = result_data['journey_time']
    algorithm_time = result_data['algorithm_time']
    path_details = result_data['path']
    
    # Extract vertex sequence from path
    path_vertices = extract_path_vertices(path_details)
    
    print(f"Visualizing path: {source} -> {destination}")
    print(f"Journey time: {journey_time} units, Algorithm time: {algorithm_time:.4f}s")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw the base network (light gray)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='lightgray', width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=30, alpha=0.6)
    
    # Track legend elements
    legend_elements = []
    
    # Highlight source and destination
    if source in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='green', 
                             node_size=200, alpha=0.8)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='green', markersize=10, label='Source'))
    
    if destination in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color='red', 
                             node_size=200, alpha=0.8)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='red', markersize=10, label='Destination'))
    
    # Highlight the shortest path
    path_edges_found = []
    if len(path_vertices) > 1:
        for i in range(len(path_vertices) - 1):
            u, v = path_vertices[i], path_vertices[i + 1]
            if G.has_edge(u, v):
                path_edges_found.append((u, v))
            else:
                # Add missing edge for visualization
                G.add_edge(u, v)
                pos[u] = pos.get(u, (np.random.random(), np.random.random()))
                pos[v] = pos.get(v, (np.random.random(), np.random.random()))
                path_edges_found.append((u, v))
        
        if path_edges_found:
            # Draw path edges in red
            nx.draw_networkx_edges(G, pos, edgelist=path_edges_found, edge_color='red', 
                                 width=4, alpha=0.9)
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3, label='Shortest Path'))
            
            # Draw path vertices in orange
            path_nodes = list(set(path_vertices))
            nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='orange', 
                                 node_size=150, alpha=0.9)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='orange', markersize=8, label='Path Nodes'))
    
    # Add labels for important nodes only
    important_nodes = set([source, destination] + path_vertices[:5])
    important_nodes = {n for n in important_nodes if n in G.nodes()}
    
    if important_nodes:
        labels_dict = {n: n for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels_dict, 
                              font_size=8, font_weight='bold', font_color='black')
    
    # Set title and layout
    plt.title(f'Shortest Path: {source} to {destination}\n'
              f'Journey Time: {journey_time} units | Algorithm Time: {algorithm_time:.4f}s',
              fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'shortest_path_{source}_to_{destination}_{timestamp}.png'
        plt.savefig(f'Visualisation/{filename}', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {filename}")
    
    plt.show()

def create_algorithm_performance_chart(result_data):
    """Create a chart showing algorithm performance metrics"""
    plt.figure(figsize=(10, 6))
    
    # Create performance metrics
    metrics = ['Journey Time', 'Algorithm Time (s)', 'Path Hops']
    values = [
        result_data['journey_time'] if result_data['journey_time'] is not None else 0,
        result_data['algorithm_time'],
        len(extract_path_vertices(result_data['path'])) - 1 if result_data['path'] else 0
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title(f'Performance: {result_data["source"]} to {result_data["destination"]}',
              fontsize=14, fontweight='bold')
    
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, f'{v:.4f}' if isinstance(v, float) else str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'performance_metrics_{timestamp}.png'
    plt.savefig(f'Visualisation/{filename}', dpi=300, bbox_inches='tight')
    print(f"Performance chart saved: {filename}")
    
    plt.show()

def main():
    """Main function for visualization"""
    parser = argparse.ArgumentParser(description='Visualize shortest path results')
    parser.add_argument('--result-file', default='shortest_path_result.json',
                       help='Path to the result JSON file')
    parser.add_argument('--csv-file', default='Datasets/network_temporal_day.csv',
                       help='Path to the original CSV data file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to file')
    
    args = parser.parse_args()
    
    # Load results
    result_data = load_path_results(args.result_file)
    if not result_data:
        return
    
    # Create visualizations
    visualize_shortest_path(result_data, args.csv_file, save_plot=not args.no_save)
    create_algorithm_performance_chart(result_data)

if __name__ == "__main__":
    main()