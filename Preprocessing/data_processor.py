import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import json
from ESD_Graph.esd_transformer import transform_temporal_to_esd

def process_network_data(csv_file_path: str, output_file: str = None, num_rows: int = None):
    """
    Process the network temporal data from CSV and convert to ESD graph.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Optional path to save the ESD graph data
        num_rows: Optional limit on number of rows to process
    
    Returns:
        ESD_graph object
    """
    # Load the CSV data
    try:
        if num_rows:
            df = pd.read_csv(csv_file_path, nrows=num_rows)
            print(f"Loaded {num_rows} records from dataset")
        else:
            df = pd.read_csv(csv_file_path)
            print(f"Loaded {len(df)} records from dataset")
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Convert to temporal edges format
    temporal_edges_list = []
    skipped_count = 0
    
    for index, row in df.iterrows():
        # Calculate duration
        duration = row['arr_time_ut'] - row['dep_time_ut']
        
        # Skip invalid durations
        if duration <= 0:
            skipped_count += 1
            continue
        
        # Create temporal edge tuple (u, v, t, λ)
        edge_tuple = (
            str(row['from_stop_I']),  # source vertex
            str(row['to_stop_I']),    # destination vertex
            int(row['dep_time_ut']),   # departure time
            int(duration)              # travel duration
        )
        temporal_edges_list.append(edge_tuple)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid edges")
    
    # Transform to ESD Graph
    esd_graph = transform_temporal_to_esd(temporal_edges_list)
    
    # Save data if output file specified
    if output_file:
        print(f"Saving ESD graph data to: {output_file}")
        save_esd_graph_data(esd_graph, temporal_edges_list, output_file)
    
    print(f"ESD Graph created with {len(esd_graph.nodes)} nodes and {sum(len(neighbors) for neighbors in esd_graph.adj.values())} edges")
    
    return esd_graph

def save_esd_graph_data(esd_graph, temporal_edges_list, output_file):
    """Save ESD graph data to JSON file for later use"""
    # Prepare data for JSON serialization
    graph_data = {
        'nodes': {},
        'adjacency': {},
        'levels': esd_graph.levels,
        'temporal_edges': temporal_edges_list
    }
    
    # Convert nodes to serializable format
    for node_id, node in esd_graph.nodes.items():
        graph_data['nodes'][node_id] = {
            'original_edge_id': node.original_edge_id,
            'u': node.u,
            'v': node.v,
            't': node.t,
            'a': node.a
        }
    
    # Convert adjacency list to serializable format
    for node_id, neighbors in esd_graph.adj.items():
        graph_data['adjacency'][node_id] = list(neighbors)
    
    with open(output_file, 'w') as f:
        json.dump(graph_data, f, indent=2)

def main():
    """Main function to run the data processor"""
    # Configuration
    CSV_FILE = "Datasets/network_temporal_day.csv"
    OUTPUT_FILE = "esd_graph_data.json"
    NUM_ROWS = None  # Set to None for full dataset, or a number to limit
    
    # Process the data
    esd_graph = process_network_data(CSV_FILE, OUTPUT_FILE, NUM_ROWS)
    
    if esd_graph:
        print("\n--- Sample ESD Graph Structure ---")
        print(esd_graph)

if __name__ == "__main__":
    main()