#!/usr/bin/env python3
"""
Generate trace file for FatTree topology.
Trace file format:
First line: number of nodes to trace
Following lines: node IDs to trace (one per line or space-separated)
"""

def generate_trace_file():
    """
    Generate trace file for FatTree topology (376 nodes).
    Trace all nodes for comprehensive monitoring.
    """
    TOTAL_NODES = 376
    
    with open("mix/trace_fattree.txt", "w") as f:
        # First line: number of nodes to trace
        f.write(f"{TOTAL_NODES}\n")
        
        # Second line: all node IDs (space-separated)
        node_ids = [str(i) for i in range(TOTAL_NODES)]
        f.write(" ".join(node_ids) + "\n")
    
    print(f"Generated trace file for {TOTAL_NODES} nodes: mix/trace_fattree.txt")

if __name__ == "__main__":
    generate_trace_file()

