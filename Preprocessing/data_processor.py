
# process_dataset.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import io
from ESD_Graph.esd_transformer import transform_temporal_to_esd #not the transformer u think


#TODO: Use a dataset to load and work with
DATASET_STRING = """from_stop_I;to_stop_I;dep_time_ut;arr_time_ut;route_type;trip_I;seq;route_I
2421;2422;1481485740;1481485779;3;1;1;1
2422;7516;1481485779;1481485814;3;1;2;1
7516;7275;1481485814;1481485860;3;1;3;1
7275;3860;1481485860;1481485892;3;1;4;1
3860;5513;1481485892;1481485928;3;1;5;1
5513;4294;1481485928;1481485951;3;1;6;1
4294;3391;1481485951;1481485995;3;1;7;1
3391;4468;1481485995;1481486015;3;1;8;1
4468;3688;1481486015;1481486040;3;1;9;1
3688;4054;1481486040;1481486074;3;1;10;1
4054;2423;1481486074;1481486108;3;1;11;1
"""

def main():
    """
    Main function to load, process, and transform the dataset.
    """
    # --- Step 1: Load the data from the string into a pandas DataFrame ---
    print("--- Loading Dataset ---")
    # Use io.StringIO to let pandas read the string as if it were a file
    df = pd.read_csv(io.StringIO(DATASET_STRING), sep=';')
    print(f"Successfully loaded {len(df)} records from the dataset.")

    # --- Step 2: Prepare the data for the transformation function ---
    # The function needs a list of tuples in the format: (u, v, t, λ)
    # where u/v are vertices, t is departure time, and λ is duration.
    print("\n--- Preparing Temporal Edges for Transformation ---")
    
    temporal_edges_list = []
    for index, row in df.iterrows():
        # The transformation logic requires duration (λ), not arrival time.
        # We calculate it: λ = arrival_time - departure_time
        duration = row['arr_time_ut'] - row['dep_time_ut']
        
        # As per the original spec, duration λ must be a positive integer.
        if duration <= 0:
            print(f"Warning: Skipping record at index {index} with non-positive duration.")
            continue
        
        # Create the tuple. We convert stop IDs to strings for robustness,
        # as some systems have non-numeric stop IDs.
        edge_tuple = (
            str(row['from_stop_I']),
            str(row['to_stop_I']),
            int(row['dep_time_ut']),
            int(duration)
        )
        temporal_edges_list.append(edge_tuple)
    
    print(f"Prepared {len(temporal_edges_list)} valid temporal edges.")

    # --- Step 3: Call the transformation function from our library file ---
    # This imports and runs the core logic from `esd_transformer.py`
    print("\n--- Calling ESD Transformation Logic ---")
    esd_graph = transform_temporal_to_esd(temporal_edges_list)

    # --- Step 4: Display the final results ---
    print("\n" + "="*40)
    print("  RESULT: ESD Graph from Dataset  ")
    print("="*40)
    print(esd_graph)
    print("="*40)

if __name__ == "__main__":
    main()