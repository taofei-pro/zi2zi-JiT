#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import re

def find_latest_event_file(log_dir):
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
    if not event_files:
        return None
    return str(max(event_files, key=os.path.getmtime))

def get_latest_loss(event_file):
    if not event_file or not os.path.exists(event_file):
        return None
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        losses = ea.Scalars('train_loss')
        if losses:
            return losses[-1].value, len(losses)
    except Exception as e:
        print(f"[EarlyStopping] Error reading loss: {e}")
    return None, 0

def main():
    if len(sys.argv) < 3:
        print("Usage: train_with_early_stopping.py <log_dir> <patience> <min_delta> -- <training_command>")
        sys.exit(1)

    log_dir = sys.argv[1]
    patience = int(sys.argv[2])
    min_delta = float(sys.argv[3])

    training_cmd = sys.argv[5:]

    print(f"[EarlyStopping] Log directory: {log_dir}")
    print(f"[EarlyStopping] Patience: {patience} epochs, Min delta: {min_delta}")
    print(f"[EarlyStopping] Starting training...")

    os.makedirs(log_dir, exist_ok=True)

    process = subprocess.Popen(
        training_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    best_loss = float('inf')
    wait_count = 0
    last_epoch_count = 0

    while True:
        ret = process.poll()
        if ret is not None:
            print(f"\n[EarlyStopping] Training process finished with code {ret}")
            break

        time.sleep(15)

        event_file = find_latest_event_file(log_dir)
        if event_file:
            result = get_latest_loss(event_file)
            if result[0] is not None:
                current_loss, epoch_count = result

                if epoch_count > last_epoch_count:
                    if epoch_count >= 5:
                        if current_loss < best_loss - min_delta:
                            best_loss = current_loss
                            wait_count = 0
                            print(f"[EarlyStopping] Epoch {epoch_count}: New best loss = {best_loss:.6f}, wait reset")
                        else:
                            wait_count += (epoch_count - last_epoch_count)
                            print(f"[EarlyStopping] Epoch {epoch_count}: Loss = {current_loss:.6f}, wait = {wait_count}/{patience}")

                            if wait_count >= patience:
                                print(f"\n[EarlyStopping] Early stopping triggered!")
                                print(f"[EarlyStopping] No improvement for {patience} epochs. Best loss: {best_loss:.6f}")
                                process.send_signal(signal.SIGTERM)
                                time.sleep(5)
                                if process.poll() is None:
                                    process.send_signal(signal.SIGKILL)
                                break

                    last_epoch_count = epoch_count

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                print(line, end='')
        except:
            pass

    sys.exit(process.returncode if process.returncode is not None else 0)

if __name__ == "__main__":
    main()
