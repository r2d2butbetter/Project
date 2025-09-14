#!/usr/bin/env python3
"""
Temporal Network Analysis Pipeline

This script provides a complete pipeline for temporal network analysis with the following features:
1. Data Processing: Load CSV data and convert to ESD graph
2. Path Finding: Find shortest paths between vertices  
3. Visualization: Generate network graphs and performance charts

Usage Examples:
    # Find specific path with visualization
    python main.py 2421 2422
    
    # Find all paths from source (no visualization)  
    python main.py 2421 --all-destinations
    
    # Use processed data to skip data loading
    python main.py 2421 2422 --skip-processing
    
    # Process limited data for faster execution
    python main.py 2421 2422 --limit 1000
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import argparse
import time
from pathlib import Path

def run_single_path_pipeline(source: str, destination: str, data_limit: int = None, skip_processing: bool = False):
    """
    Run the complete pipeline for a specific source-destination pair with visualization.
    
    Args:
        source: Source vertex ID
        destination: Destination vertex ID  
        data_limit: Limit number of rows to process (for faster execution)
        skip_processing: Skip data processing step if ESD graph already exists
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Finding path from {source} to {destination}")
    
    # Step 1: Data Processing (if not skipped)
    if not skip_processing:
        try:
            from Preprocessing.data_processor import process_network_data
            esd_graph = process_network_data(
                "Datasets/network_temporal_day.csv", 
                "esd_graph_data.json", 
                data_limit
            )
            if not esd_graph:
                print("Data processing failed")
                return False
        except Exception as e:
            print(f"Data processing failed: {e}")
            return False
    
    # Step 2: Path Finding
    try:
        from find_shortest_path import find_shortest_path
        
        use_json_data = Path("esd_graph_data.json").exists()
        data_source = "esd_graph_data.json" if use_json_data else "Datasets/network_temporal_day.csv"
        
        journey_time, path, algorithm_time = find_shortest_path(
            source, 
            destination, 
            data_source,
            use_saved_data=use_json_data
        )
        
        if journey_time is None:
            print("No path found between the specified vertices")
            return False
            
        print(f"Path found - Journey time: {journey_time} units, Algorithm time: {algorithm_time:.4f}s")
        
    except Exception as e:
        print(f"Path finding failed: {e}")
        return False
    
    # Step 3: Visualization
    try:
        os.makedirs("Visualisation", exist_ok=True)
        
        from Visualisation.path_visualizer import load_path_results, visualize_shortest_path, create_algorithm_performance_chart
        
        result_data = load_path_results()
        if result_data:
            if result_data.get('source') != source or result_data.get('destination') != destination:
                print(f"Warning: Result mismatch - expected {source} to {destination}")
                return False
            
            visualize_shortest_path(result_data, "Datasets/network_temporal_day.csv")
            create_algorithm_performance_chart(result_data)
            print("Visualization completed")
        else:
            print("Failed to load results for visualization")
            return False
            
    except Exception as e:
        print(f"Visualization failed: {e}")
        return False
    
    return True

def run_all_destinations_pipeline(source: str, data_limit: int = None):
    """
    Find fastest paths from source to ALL destinations.
    
    Args:
        source: Source vertex ID
        data_limit: Limit number of rows to process
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Finding all destinations from source: {source}")
    
    try:
        from Preprocessing.data_processor import process_network_data
        esd_graph = process_network_data(
            "Datasets/network_temporal_day.csv", 
            output_file=None,
            num_rows=data_limit
        )
        
        if not esd_graph:
            print("Data processing failed")
            return False
        
        from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
        fpd_solver = SerialESDG_FPD(esd_graph)
        
        print(f"Running FPD algorithm from source: {source}")
        start_time = time.time()
        journey_times, fastest_paths = fpd_solver.find_fastest_paths(source)
        end_time = time.time()
        algorithm_time = end_time - start_time
        
        # Display results
        sorted_results = sorted(journey_times.items(), key=lambda item: item[1])
        
        print(f"\n{'Destination':<15} | {'Journey Duration':<20}")
        print("-" * 40)
        
        reachable_count = 0
        for dest, journey_time in sorted_results:
            if journey_time == float('inf'):
                status = "Unreachable"
            elif dest == source:
                status = "Source"
            else:
                reachable_count += 1
                status = "Reachable"
                
            print(f"{dest:<15} | {journey_time:<20}")
        
        print(f"\nAlgorithm execution time: {algorithm_time:.4f} seconds")
        print(f"Reachable destinations: {reachable_count}/{len(journey_times)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Temporal Network Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2421 2422                    # Find specific path with visualization
  %(prog)s 2421 --all-destinations      # Find all paths from source
  %(prog)s 2421 2422 --skip-processing  # Use existing processed data
  %(prog)s 2421 2422 --limit 1000       # Process limited data for speed
        """
    )
    
    parser.add_argument('source', help='Source vertex ID')
    parser.add_argument('destination', nargs='?', help='Destination vertex ID (required unless --all-destinations)')
    parser.add_argument('--all-destinations', action='store_true',
                       help='Find paths to all destinations from source (no visualization)')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of data rows to process (for faster execution)')
    parser.add_argument('--skip-processing', action='store_true', 
                       help='Skip data processing step (use existing processed data)')
    parser.add_argument('--force-fresh', action='store_true',
                       help='Delete existing result files to force fresh computation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_destinations and not args.destination:
        parser.error("Destination is required unless --all-destinations is specified")
    elif args.all_destinations and args.destination:
        parser.error("Cannot specify both destination and --all-destinations")
    
    # Clean up old results if requested
    if args.force_fresh:
        result_file = Path("shortest_path_result.json")
        if result_file.exists():
            result_file.unlink()
            print("Deleted existing result file")
    
    # Run appropriate pipeline
    if args.all_destinations:
        success = run_all_destinations_pipeline(args.source, args.limit)
    else:
        success = run_single_path_pipeline(
            args.source, 
            args.destination, 
            args.limit, 
            args.skip_processing
        )
    
    if not success:
        print("Pipeline failed")
        sys.exit(1)
    else:
        print("Pipeline completed successfully")

if __name__ == "__main__":
    main()