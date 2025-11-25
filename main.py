import pandas as pd
import logging
import time
from ESD_Graph.esd_transformer import transform_temporal_to_esd

# Import all three solvers
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from FPD_Algorithm.parallel_esdg_fpd import ParallelESDG_FPD  # Algorithm 1 (MBFS)
from FPD_Algorithm.parallel_esdg_lo import ParallelESDG_LO    # Algorithm 3 (Level Order)

from utils.graph_caching import save_esd_graph_to_json, load_esd_graph_from_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(dataset_path: str, source_node: str, num_rows: int = None):
    """
    Executes the full pipeline: Load -> Transform -> Compute FPD -> Compare.
    """
    # --- STEP 1: LOAD DATA ---
    print("="*60); print("STEP 1: LOADING DATA"); print("="*60)
    df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    temporal_edges_list = [(str(row['from_stop_I']), str(row['to_stop_I']),
                            int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
                           for _, row in df.iterrows() if row['arr_time_ut'] - row['dep_time_ut'] > 0]
    print(f"Loaded {len(temporal_edges_list)} temporal edges.")

    # --- STEP 2: BUILD ESD GRAPH ---
    print("\n" + "="*60); print("STEP 2: ESD GRAPH GENERATION"); print("="*60)
    esd_graph = load_esd_graph_from_json(num_rows)
    if esd_graph is None:
        esd_graph = transform_temporal_to_esd(temporal_edges_list)
        save_esd_graph_to_json(esd_graph, num_rows)
    
    # Vertex Validation
    all_vertices = set()
    for node in esd_graph.nodes.values():
        all_vertices.add(int(node.u)); all_vertices.add(int(node.v))
    
    source_int = int(source_node)
    if source_int not in all_vertices:
        if len(all_vertices) > 0:
            alt_source = sorted(list(all_vertices))[0]
            print(f"⚠️ Source {source_node} not found. Switching to {alt_source}.")
            source_node = str(alt_source)
        else:
            print("❌ Graph is empty."); return

    print("\n" + "="*60); print(f"STEP 3: RUNNING & COMPARING ALGORITHMS (Source: {source_node})"); print("="*60)

    results = {} # To store times and outputs

    # ---------------------------------------------------------
    # 1. SERIAL CPU
    # ---------------------------------------------------------
    print("\n--- 1. Serial CPU ---")
    t_start = time.perf_counter()
    solver_serial = SerialESDG_FPD(esd_graph)
    res_serial, paths_serial = solver_serial.find_fastest_paths(source_node)
    t_serial = time.perf_counter() - t_start
    print(f"⏱️ Time: {t_serial:.4f}s")
    results['Serial'] = {'time': t_serial, 'data': res_serial}

    # ---------------------------------------------------------
    # 2. PARALLEL GPU: Algorithm 1 (MBFS)
    # ---------------------------------------------------------
    print("\n--- 2. Parallel GPU (Algo 1: MBFS) ---")
    t_init_start = time.perf_counter()
    solver_mbfs = ParallelESDG_FPD(esd_graph)
    t_init_mbfs = time.perf_counter() - t_init_start
    
    t_compute_start = time.perf_counter()
    res_mbfs, _ = solver_mbfs.find_fastest_paths(source_node, reconstruct_paths=False)
    t_compute_mbfs = time.perf_counter() - t_compute_start
    
    print(f"⏱️ Init: {t_init_mbfs:.4f}s | Compute: {t_compute_mbfs:.4f}s | Total: {t_init_mbfs + t_compute_mbfs:.4f}s")
    results['MBFS'] = {'time': t_compute_mbfs, 'total': t_init_mbfs + t_compute_mbfs, 'data': res_mbfs}

    # ---------------------------------------------------------
    # 3. PARALLEL GPU: Algorithm 3 (Level Order)
    # ---------------------------------------------------------
    print("\n--- 3. Parallel GPU (Algo 3: Level Order) ---")
    t_init_start = time.perf_counter()
    solver_lo = ParallelESDG_LO(esd_graph)
    t_init_lo = time.perf_counter() - t_init_start
    
    t_compute_start = time.perf_counter()
    res_lo, _ = solver_lo.find_fastest_paths(source_node, reconstruct_paths=False)
    t_compute_lo = time.perf_counter() - t_compute_start
    
    print(f"⏱️ Init: {t_init_lo:.4f}s | Compute: {t_compute_lo:.4f}s | Total: {t_init_lo + t_compute_lo:.4f}s")
    results['LO'] = {'time': t_compute_lo, 'total': t_init_lo + t_compute_lo, 'data': res_lo}

    # ---------------------------------------------------------
    # 4. COMPARISON & VALIDATION
    # ---------------------------------------------------------
    print("\n" + "="*60); print("PERFORMANCE SUMMARY"); print("="*60)
    
    # Calculate Speedups relative to Serial
    speedup_mbfs = t_serial / results['MBFS']['time']
    speedup_lo = t_serial / results['LO']['time']
    
    print(f"{'Algorithm':<20} | {'Compute Time':<15} | {'Speedup vs Serial':<20} | {'Match?':<10}")
    print("-" * 75)
    
    # --- Robust Validation Logic ---
    # 1. Filter out 'inf' from Serial results. GPU results only contain reachable nodes.
    #    Comparing {A:10, B:inf} with {A:10} should pass.
    serial_clean = {k: v for k, v in results['Serial']['data'].items() if v != float('inf')}
    
    # 2. Check MBFS
    mbfs_data = results['MBFS']['data']
    # Check if lengths match and if intersection of items matches the clean serial set
    match_mbfs = (len(mbfs_data) == len(serial_clean)) and \
                 (len(set(mbfs_data.items()) & set(serial_clean.items())) == len(serial_clean))
    
    print(f"{'Serial CPU':<20} | {t_serial:<15.4f} | {'1.00x':<20} | {'N/A':<10}")
    print(f"{'Parallel MBFS':<20} | {results['MBFS']['time']:<15.4f} | {speedup_mbfs:<20.2f} | {'✅' if match_mbfs else '❌'}")
    
    # 3. Check LO
    lo_data = results['LO']['data']
    match_lo = (len(lo_data) == len(serial_clean)) and \
               (len(set(lo_data.items()) & set(serial_clean.items())) == len(serial_clean))
    
    print(f"{'Parallel LO':<20} | {results['LO']['time']:<15.4f} | {speedup_lo:<20.2f} | {'✅' if match_lo else '❌'}")

    # Debug info if LO mismatch
    if not match_lo:
        print("\n🔍 LO Mismatch Debug:")
        print(f"   Serial (Clean) Count: {len(serial_clean)}")
        print(f"   LO Count: {len(lo_data)}")
        
        # Check for missing keys
        missing = set(serial_clean.keys()) - set(lo_data.keys())
        extra = set(lo_data.keys()) - set(serial_clean.keys())
        if missing: print(f"   Missing Keys in LO: {list(missing)[:5]}...")
        if extra: print(f"   Extra Keys in LO: {list(extra)[:5]}...")
        
        # Check value mismatches
        common = set(serial_clean.keys()) & set(lo_data.keys())
        mismatches = [(k, serial_clean[k], lo_data[k]) for k in common if serial_clean[k] != lo_data[k]]
        if mismatches:
            print(f"   Value Mismatches (first 5): {mismatches[:5]}")

    print("\n💡 NOTE:")
    print("   - MBFS (Algo 1): Overhead per phase. Slow on small data.")
    print("   - LO (Algo 3): Single pass. Fast on small data.")

    # ---------------------------------------------------------
    # 5. SAMPLE PATH OUTPUT
    # ---------------------------------------------------------
    print("\n" + "="*60); print(f"SAMPLE OUTPUT (Serial Reconstructed Paths)"); print("="*60)
    sorted_paths = sorted(results['Serial']['data'].items(), key=lambda x: x[1])[:10]
    
    print(f"{'Dest':<10} | {'Duration':<10} | {'Path'}")
    for dest, duration in sorted_paths:
        path_nodes = paths_serial.get(dest)
        path_str = " -> ".join([f"e{n.original_edge_id}" for n in path_nodes]) if path_nodes else "N/A"
        if dest == source_node: path_str = "Source"
        print(f"{dest:<10} | {duration:<10} | {path_str}")

if __name__ == "__main__":
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "3391" 
    NUM_ROWS = 100000 
    
    run_pipeline(DATASET_FILE, SOURCE_VERTEX, NUM_ROWS)