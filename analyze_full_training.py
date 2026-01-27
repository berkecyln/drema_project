import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_path = "/home/gunnleif/Projects/drema_project/logs/Experiment_20260127-175850/gold_data_30k_loss_tuned"
files = [f for f in os.listdir(log_path) if "tfevents" in f]
if not files: exit()
file_path = os.path.join(log_path, files[0])

event_acc = EventAccumulator(file_path)
event_acc.Reload()

# Try to find the total loss tag
tags = event_acc.Tags()['scalars']
target_tag = 'train_loss_patches/total_loss'
if target_tag not in tags:
    candidates = [t for t in tags if 'loss' in t and 'total' in t]
    target_tag = candidates[0] if candidates else tags[0]

print(f"Analyzing Tag: {target_tag}\n")
print(f"{ 'Step':<10} | { 'Loss':<10} | {'Delta'}")
print("-" * 35)

prev_loss = None
events = event_acc.Scalars(target_tag)

# Sample every ~1000 steps and critical points
critical_points = [6000, 7000, 15000, 16000, 29000]

for i, event in enumerate(events):
    step = event.step
    loss = event.value
    
    # Print every 1000 steps OR at critical transitions
    if step % 1000 < 100 or any(abs(step - cp) < 100 for cp in critical_points):
        # Simple debouncing to avoid printing 5 lines for step 15001, 15002...
        if i % 10 == 0: 
            delta_str = ""
            if prev_loss:
                delta = loss - prev_loss
                if delta > 0.01: delta_str = f"(+{delta:.4f}) !!! SPIKE"
                elif delta > 0: delta_str = f"(+{delta:.4f})"
                else: delta_str = f"({delta:.4f})"
            
            print(f"{step:<10} | {loss:.6f}   | {delta_str}")
            prev_loss = loss
