#!/usr/bin/env python3
"""
Generate FatTree topology for Fusion experiments according to paper specifications:
- 16 core switches
- 20 aggregation switches  
- 20 ToR switches
- Each ToR connects to 16 servers (25 Gbps links)
- All inter-switch connections are 100 Gbps links
- Each link has 1 microsecond delay
"""

def generate_fattree_topology():
    """
    Generate FatTree topology file according to paper specifications.
    
    Topology structure:
    - Core switches: 0-15 (16 switches)
    - Aggregation switches: 16-35 (20 switches)
    - ToR switches: 36-55 (20 switches)
    - Servers: 56-375 (320 servers, 20 ToR × 16 servers each)
    
    Total: 376 nodes (320 servers + 56 switches)
    """
    
    # Node IDs
    NUM_CORE = 16
    NUM_AGG = 20
    NUM_TOR = 20
    SERVERS_PER_TOR = 16
    
    CORE_START = 0
    CORE_END = CORE_START + NUM_CORE - 1
    
    AGG_START = CORE_END + 1
    AGG_END = AGG_START + NUM_AGG - 1
    
    TOR_START = AGG_END + 1
    TOR_END = TOR_START + NUM_TOR - 1
    
    SERVER_START = TOR_END + 1
    SERVER_END = SERVER_START + NUM_TOR * SERVERS_PER_TOR - 1
    
    TOTAL_NODES = SERVER_END + 1
    TOTAL_SWITCHES = NUM_CORE + NUM_AGG + NUM_TOR
    
    links = []
    
    # 1. Connect servers to ToR switches (25 Gbps, 1us delay)
    # Each ToR switch connects to 16 servers
    for tor_idx in range(NUM_TOR):
        tor_id = TOR_START + tor_idx
        for server_idx in range(SERVERS_PER_TOR):
            server_id = SERVER_START + tor_idx * SERVERS_PER_TOR + server_idx
            links.append((server_id, tor_id, "25Gbps", "0.001ms", 0))
    
    # 2. Connect ToR switches to Aggregation switches (100 Gbps, 1us delay)
    # Each ToR connects to multiple aggregation switches
    # For a balanced FatTree: each ToR connects to 4 aggregation switches
    # Each aggregation switch connects to 4 ToR switches
    for tor_idx in range(NUM_TOR):
        tor_id = TOR_START + tor_idx
        # Each ToR connects to 4 aggregation switches
        agg_group = tor_idx // 4  # Group ToR switches
        for agg_offset in range(4):
            agg_id = AGG_START + agg_group * 4 + agg_offset
            if agg_id <= AGG_END:
                links.append((tor_id, agg_id, "100Gbps", "0.001ms", 0))
    
    # 3. Connect Aggregation switches to Core switches (100 Gbps, 1us delay)
    # Each aggregation switch connects to multiple core switches
    # For a balanced FatTree: each aggregation connects to 4 core switches
    # Each core switch connects to 5 aggregation switches
    for agg_idx in range(NUM_AGG):
        agg_id = AGG_START + agg_idx
        # Each aggregation connects to 4 core switches
        for core_offset in range(4):
            core_id = CORE_START + (agg_idx * 4 + core_offset) % NUM_CORE
            links.append((agg_id, core_id, "100Gbps", "0.001ms", 0))
    
    # Write topology file
    with open("mix/fattree_topology.txt", "w") as f:
        # First line: total nodes, switch nodes, total links
        f.write(f"{TOTAL_NODES} {TOTAL_SWITCHES} {len(links)}\n")
        
        # Second line: switch node IDs (all switches)
        switch_ids = []
        for i in range(CORE_START, CORE_END + 1):
            switch_ids.append(str(i))
        for i in range(AGG_START, AGG_END + 1):
            switch_ids.append(str(i))
        for i in range(TOR_START, TOR_END + 1):
            switch_ids.append(str(i))
        f.write(" ".join(switch_ids) + "\n")
        
        # Remaining lines: links (src dst rate delay error_rate)
        for src, dst, rate, delay, error in links:
            f.write(f"{src} {dst} {rate} {delay} {error}\n")
    
    print(f"Generated FatTree topology:")
    print(f"  Total nodes: {TOTAL_NODES}")
    print(f"  Total switches: {TOTAL_SWITCHES}")
    print(f"    - Core switches: {NUM_CORE} (IDs {CORE_START}-{CORE_END})")
    print(f"    - Aggregation switches: {NUM_AGG} (IDs {AGG_START}-{AGG_END})")
    print(f"    - ToR switches: {NUM_TOR} (IDs {TOR_START}-{TOR_END})")
    print(f"  Total servers: {NUM_TOR * SERVERS_PER_TOR} (IDs {SERVER_START}-{SERVER_END})")
    print(f"  Total links: {len(links)}")
    print(f"  Topology file: mix/fattree_topology.txt")

if __name__ == "__main__":
    generate_fattree_topology()

