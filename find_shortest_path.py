import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import argparse
import time
import json
from Preprocessing.data_processor import process_network_data
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from ESD_Graph.structures.esd_graph import ESD_graph, ESD_Node

def load_esd_graph_from_file(json_file):
    """Load ESD graph from JSON file"""
    with open(json_file, 'r') as f:
        graph_data = json.load(f)
    
    esd_graph = ESD_graph()
    
    # Add nodes
    for node_id, node_data in graph_data['nodes'].items():
        node = ESD_Node(
            original_edge_id=node_data['original_edge_id'],
            u=node_data['u'],
            v=node_data['v'],
            t=node_data['t'],
            a=node_data['a']
        )
        esd_graph.add_node(node)
    
    # Add edges
    for node_id, neighbors in graph_data['adjacency'].items():
        for neighbor_id in neighbors:
            esd_graph.add_edge(int(node_id), int(neighbor_id))
    
    # Set levels
    esd_graph.levels = {int(k): v for k, v in graph_data['levels'].items()}
    
    return esd_graph

def save_path_results(source, destination, journey_time, path, algorithm_time, result_file='shortest_path_result.json'):
    """Save path finding results to JSON file"""
    result_data = {
        'source': source,
        'destination': destination,
        'journey_time': journey_time,
        'algorithm_time': algorithm_time,
        'path': [str(node) for node in path] if path else [],
        'timestamp': time.time()
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

def find_shortest_path(source: str, destination: str, data_source: str = None, use_saved_data: bool = False):
    """
    Find shortest path between source and destination.
    
    Args:
        source: Source vertex ID
        destination: Destination vertex ID
        data_source: Path to CSV file or JSON file
        use_saved_data: Whether to use saved JSON data or process from CSV
    
    Returns:
        Tuple of (journey_time, path, algorithm_time)
    """
    # Load or create ESD graph
    if use_saved_data and data_source and data_source.endswith('.json'):
        esd_graph = load_esd_graph_from_file(data_source)
    else:
        # Process from CSV
        csv_file = data_source or "Datasets/network_temporal_day.csv"
        esd_graph = process_network_data(csv_file) 
        
        if not esd_graph:
            print("Failed to create ESD graph")
            return None, None, None
    
    fpd_solver = SerialESDG_FPD(esd_graph)
    
    # Measure algorithm execution time
    start_time = time.perf_counter()
    journey_times, fastest_paths = fpd_solver.find_fastest_paths(source)
    end_time = time.perf_counter()
    algorithm_time = end_time - start_time
    
    # Extract results for destination
    if destination in journey_times:
        journey_time = journey_times[destination]
        path = fastest_paths.get(destination, [])
        
        print(f"Path found: {source} -> {destination}")
        print(f"Journey time: {journey_time} units, Algorithm time: {algorithm_time:.4f}s")
        
        # Save results
        save_path_results(source, destination, journey_time, path, algorithm_time)
        
        return journey_time, path, algorithm_time
    
    else:
        print(f"No path found from {source} to {destination}")
        print(f"Algorithm time: {algorithm_time:.4f}s")
        
        # Save "no path found" results
        save_path_results(source, destination, None, [], algorithm_time)
        
        return None, None, algorithm_time

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Find shortest path in temporal network')
    parser.add_argument('source', help='Source vertex ID')
    parser.add_argument('destination', help='Destination vertex ID')
    parser.add_argument('--data', help='Path to data file (CSV or JSON)', default="Datasets/network_temporal_day.csv")
    parser.add_argument('--use-saved', action='store_true', help='Use saved JSON data instead of processing CSV')
    
    args = parser.parse_args()
    
    journey_time, path, algorithm_time = find_shortest_path(
        args.source, 
        args.destination, 
        args.data, 
        args.use_saved
    )

if __name__ == "__main__":
    main()