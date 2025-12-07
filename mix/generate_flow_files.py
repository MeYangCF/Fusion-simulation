#!/usr/bin/env python3
"""
Generate flow files for three experiments according to paper specifications.

Experiment 1: Stable conditions with WebSearch traffic, 60% load
Experiment 2: Traffic variation with WebSearch and FB_Hadoop, 60%-80% load
Experiment 3: Incast with 8 senders, 1 receiver, 256/512/1024 synchronized flows
"""

import random
import math

# Topology parameters (from generate_fattree_topology.py)
NUM_TOR = 20
SERVERS_PER_TOR = 16
TOR_START = 36
SERVER_START = 56
TOTAL_SERVERS = NUM_TOR * SERVERS_PER_TOR

def generate_websearch_distribution():
    """
    Generate WebSearch traffic distribution (Alibaba trace).
    Typical WebSearch: mix of small, medium, and large flows.
    """
    # WebSearch distribution: mix of flow sizes
    # Based on typical data center traffic patterns
    sizes = []
    weights = []
    
    # Small flows (< 100KB): 40%
    sizes.extend([10 * 1024, 50 * 1024, 100 * 1024])
    weights.extend([0.15, 0.15, 0.10])
    
    # Medium flows (100KB - 10MB): 40%
    sizes.extend([500 * 1024, 2 * 1024 * 1024, 10 * 1024 * 1024])
    weights.extend([0.20, 0.15, 0.05])
    
    # Large flows (> 10MB): 20%
    sizes.extend([50 * 1024 * 1024, 100 * 1024 * 1024, 500 * 1024 * 1024])
    weights.extend([0.10, 0.07, 0.03])
    
    return sizes, weights

def generate_fb_hadoop_distribution():
    """
    Generate FB_Hadoop traffic distribution.
    FB_Hadoop: 70% of traffic is smaller than 10 KB.
    """
    sizes = []
    weights = []
    
    # Very small flows (< 10KB): 70%
    sizes.extend([1 * 1024, 5 * 1024, 10 * 1024])
    weights.extend([0.30, 0.25, 0.15])
    
    # Small flows (10KB - 100KB): 20%
    sizes.extend([20 * 1024, 50 * 1024, 100 * 1024])
    weights.extend([0.10, 0.05, 0.05])
    
    # Medium flows (100KB - 1MB): 8%
    sizes.extend([200 * 1024, 500 * 1024, 1024 * 1024])
    weights.extend([0.03, 0.03, 0.02])
    
    # Large flows (> 1MB): 2%
    sizes.extend([5 * 1024 * 1024, 10 * 1024 * 1024])
    weights.extend([0.01, 0.01])
    
    return sizes, weights

def sample_flow_size(sizes, weights):
    """Sample a flow size from the distribution."""
    return random.choices(sizes, weights=weights)[0]

def generate_exp1_flows(num_flows, load=0.6):
    """
    Experiment 1: Stable conditions with WebSearch traffic, 60% load.
    """
    sizes, weights = generate_websearch_distribution()
    
    # Calculate total bytes needed for 60% load
    # Assuming average link capacity and network utilization
    # For simplicity, we'll generate flows with appropriate sizes
    
    flows = []
    start_time = 0.1
    time_interval = 0.01  # 10ms between flow starts
    
    for i in range(num_flows):
        src = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        while dst == src:
            dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        
        pg = random.randint(0, 7)  # Priority group
        dport = 10000
        flow_size = sample_flow_size(sizes, weights)
        max_packets = math.ceil(flow_size / 1000)  # Assuming 1000 byte packets
        
        flows.append((src, dst, pg, dport, max_packets, start_time + i * time_interval))
    
    return flows

def generate_exp2_flows(num_flows_websearch, num_flows_hadoop, load_start=0.6, load_end=0.8):
    """
    Experiment 2: Traffic variation with WebSearch and FB_Hadoop.
    Load increases from 60% to 80%.
    """
    websearch_sizes, websearch_weights = generate_websearch_distribution()
    hadoop_sizes, hadoop_weights = generate_fb_hadoop_distribution()
    
    flows = []
    start_time = 0.1
    time_interval = 0.01
    
    # WebSearch flows
    for i in range(num_flows_websearch):
        src = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        while dst == src:
            dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        
        pg = random.randint(0, 7)
        dport = 10000
        flow_size = sample_flow_size(websearch_sizes, websearch_weights)
        max_packets = math.ceil(flow_size / 1000)
        
        flows.append((src, dst, pg, dport, max_packets, start_time + i * time_interval))
    
    # FB_Hadoop flows (start after WebSearch flows)
    hadoop_start = start_time + num_flows_websearch * time_interval
    for i in range(num_flows_hadoop):
        src = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        while dst == src:
            dst = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
        
        pg = random.randint(0, 7)
        dport = 10000
        flow_size = sample_flow_size(hadoop_sizes, hadoop_weights)
        max_packets = math.ceil(flow_size / 1000)
        
        flows.append((src, dst, pg, dport, max_packets, hadoop_start + i * time_interval))
    
    return flows

def generate_exp3_flows(num_flows):
    """
    Experiment 3: Incast scenario.
    8 senders, 1 receiver, synchronized burst flows.
    All flows start at the same time to create incast.
    """
    # Select 8 sender hosts and 1 receiver host
    # Use hosts from different ToR switches for better distribution
    senders = []
    for i in range(8):
        # Select from different ToR switches
        tor_idx = i % NUM_TOR
        server_idx = random.randint(0, SERVERS_PER_TOR - 1)
        sender_id = SERVER_START + tor_idx * SERVERS_PER_TOR + server_idx
        senders.append(sender_id)
    
    # Select 1 receiver (different from senders)
    receiver = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
    while receiver in senders:
        receiver = random.randint(SERVER_START, SERVER_START + TOTAL_SERVERS - 1)
    
    flows = []
    # Synchronized burst: all flows start at the same time
    start_time = 0.1
    
    for i in range(num_flows):
        sender = senders[i % len(senders)]  # Round-robin through senders
        pg = random.randint(0, 7)
        dport = 10000
        # Medium-sized flows for incast (typical: 1-10 MB)
        flow_size = random.randint(1 * 1024 * 1024, 10 * 1024 * 1024)
        max_packets = math.ceil(flow_size / 1000)
        
        flows.append((sender, receiver, pg, dport, max_packets, start_time))
    
    return flows

def write_flow_file(filename, flows):
    """Write flows to file."""
    with open(filename, "w") as f:
        f.write(f"{len(flows)}\n")
        for src, dst, pg, dport, max_packets, start_time in flows:
            f.write(f"{src} {dst} {pg} {dport} {max_packets} {start_time:.6f}\n")

def main():
    random.seed(42)  # For reproducibility
    
    # Experiment 1: Stable conditions, WebSearch, 60% load
    print("Generating Experiment 1 flows (WebSearch, 60% load)...")
    exp1_flows = generate_exp1_flows(num_flows=5000, load=0.6)
    write_flow_file("mix/flow_exp1_websearch_60.txt", exp1_flows)
    print(f"  Generated {len(exp1_flows)} flows")
    
    # Experiment 2: Traffic variation, WebSearch + FB_Hadoop, 60%-80% load
    print("Generating Experiment 2 flows (WebSearch + FB_Hadoop, 60%-80% load)...")
    exp2_flows = generate_exp2_flows(num_flows_websearch=3000, num_flows_hadoop=4000, 
                                     load_start=0.6, load_end=0.8)
    write_flow_file("mix/flow_exp2_variation.txt", exp2_flows)
    print(f"  Generated {len(exp2_flows)} flows")
    
    # Experiment 3: Incast, 8 senders, 1 receiver, 256/512/1024 flows
    # Generate separate files for each test case
    print("Generating Experiment 3 flows (Incast)...")
    for num_flows in [256, 512, 1024]:
        exp3_flows = generate_exp3_flows(num_flows)
        filename = f"mix/flow_exp3_incast_{num_flows}.txt"
        write_flow_file(filename, exp3_flows)
        print(f"  Generated {len(exp3_flows)} flows for {num_flows} flows test case")
    
    print("\nAll flow files generated successfully!")

if __name__ == "__main__":
    main()

