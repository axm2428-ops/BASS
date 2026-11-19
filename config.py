"""
Motor Imagery Neurofeedback Experiment - Main Data Collection Script

IMPORTANT: This script requires an EEG data source (OpenBCI or synthetic).
Before running this script:
  1. Start your EEG data source (BrainVision/OpenBCI)
  2. OR run test_lsl_connections.py to verify connections

To test if marker collection is working:
  python test_marker_fix.py

See MARKER_FIX_SUMMARY.md for details on the marker collection fix.
"""
import threading
import time
import random
import csv
import os
import sys
from collections import deque
from datetime import datetime
import numpy as np
import logging
import argparse
import uuid
from scipy.signal import butter, filtfilt
import sys
import config


sys.path.insert(0, '.')



# Import from current directory
from calculate_band_power import calculate_band_power
from fft_power_update import fft_power_update
from bandpass_filter import bandpass_filter
#import neurofeedback processor
from NeurofeedbackProcessor import NeurofeedbackProcessor
from run_neurofeedback_experiment import run_neurofeedback_experiment
# Import lsl data collector class
from LSLDataCollector import LSLDataCollector



# Configure logging: timestamped, INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


from config import (
    EEG_STREAM_NAME,
    MARKER_STREAM_NAME,
    SAMPLE_RATE,
    EEG_CHANNELS,
    CSV_EEG_CHANNELS,
    EEG_STREAM_NAME,
    MARKER_STREAM_NAME,
    NUM_TRIALS,
    INSTRUCTION_DURATION,
    IMAGERY_DURATION,
    INTER_TRIAL_INTERVAL,
    NO_MOVEMENT_PROBABILITY,
    MARKER_RIGHT,
    MARKER_LEFT,
    MARKER_FEET,
    MARKER_STOP,
    MARKER_NO_MOVEMENT,
    RAW_DATA_DIR,
    MERGE_THRESHOLD,
    POLL_SLEEP,
    MU_BAND,
    BETA_BAND,
    WINDOW_LENGTH,
    WINDOW_OVERLAP,
    WINDOW_SHIFT,
    BASELINE_START,
    BASELINE_END,
    INITIAL_BASELINE_DURATION,
    BASELINE_WEIGHT_INITIAL,
    BASELINE_WEIGHT_TRIAL,
    DISPLAY_REFRESH_RATE,
    DISPLAY_UPDATE_INTERVAL,
    FEEDBACK_BAR_MAX_WIDTH,
    SMOOTHING_FACTOR,
    ERD_EMPHASIS_FACTOR,
    USE_FFT_BUFFER,
    ADAPTIVE_ERD_SCALING,
    MAX_INITIAL_ERD,
    ERD_SCALING_FACTOR,
    FILTER_ORDER,
    FILTER_BAND
)

# Optional BIDS settings — fall back to sensible defaults if not present in config
try:
    from config import SUBJECT_ID, SESSION_ID, TASK_NAME, RUN_ID, BIDS_ROOT
except Exception:
    SUBJECT_ID = "sub-01"
    SESSION_ID = "ses-01"
    TASK_NAME = "mi"
    RUN_ID = "run-01"
    BIDS_ROOT = "bids_dataset"


###############################################################################
#                             BIDS CONVERSION
###############################################################################
def convert_to_bids(csv_path):
    """
    This function is given to you.
    It converts the CSV data to BIDS (Brain Imaging Data Structure) format.
    """
    logging.info(f"Converting {csv_path} to BIDS...")
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.exception("Failed to read CSV file for BIDS conversion:")
        return

    # Verify that expected EEG channel columns exist
    missing_cols = [ch for ch in CSV_EEG_CHANNELS if ch not in df.columns]
    if missing_cols:
        logging.error(f"CSV is missing expected EEG channels: {missing_cols}")
        return

    # Extract EEG data
    data_eeg = []
    for ch in CSV_EEG_CHANNELS:
        col_data = pd.to_numeric(df[ch], errors="coerce").fillna(0).values
        data_eeg.append(col_data)
    data_eeg = np.array(data_eeg)  # shape: (n_channels, n_samples)

    # Process markers and trial_id
    raw_markers = df["marker"].astype(str).replace({"nan": "", "None": ""}).values
    trial_ids = df["trial_id"].astype(str).values if "trial_id" in df.columns else ["0"] * len(df)
    markers = []
    for i, (m, t) in enumerate(zip(raw_markers, trial_ids)):
        if m != "":
            markers.append((i, m, t))

    from mne import create_info
    from mne.io import RawArray
    info = create_info(ch_names=CSV_EEG_CHANNELS, sfreq=SAMPLE_RATE, ch_types="eeg")
    raw = RawArray(data_eeg, info)

    # Create annotations from markers with trial_id
    annotations_onset = []
    annotations_desc  = []
    for (idx, marker_val, trial_id) in markers:
        onset_sec = idx / SAMPLE_RATE  # Alternatively, use LSL timestamps if available
        annotations_onset.append(onset_sec)
        annotations_desc.append(f"Marker/{marker_val}/trial_{trial_id}")

    import mne
    raw.set_annotations(mne.Annotations(
        onset=annotations_onset,
        duration=[0.0]*len(annotations_onset),
        description=annotations_desc
    ))

    from mne_bids import write_raw_bids, BIDSPath
    bids_path = BIDSPath(
        root= BIDS_ROOT,
        subject=SUBJECT_ID.replace("sub-", ""),
        session=SESSION_ID.replace("ses-", ""),
        task=TASK_NAME.replace("task-", ""),
        run=RUN_ID.replace("run-", ""),
        datatype="eeg"
    )
    try:
        write_raw_bids(
            raw,
            bids_path,
            overwrite=True,
            allow_preload=True,
            format='BrainVision'
        )
        logging.info(f"BIDS dataset written at: {bids_path.directory}")
    except Exception as e:
        logging.exception("Error during BIDS conversion:")

def main():


    logging.info("="*70)
    logging.info("Starting Motor Imagery Neurofeedback Experiment")
    logging.info("="*70)

    #import neurofeedback processor
    neurofeedback_processor = NeurofeedbackProcessor()

    # Create marker stream BEFORE starting collector to avoid race condition
    from pylsl import StreamInfo, StreamOutlet
    try:
        marker_source_id = f"mi-marker-{uuid.uuid4().hex}"
        marker_info = StreamInfo(
            MARKER_STREAM_NAME, "Markers", 1, 0, "string", marker_source_id
        )
        marker_outlet = StreamOutlet(marker_info)
        logging.info(f"✓ Marker stream created: '{MARKER_STREAM_NAME}'")
        # Brief delay to let marker stream stabilize on the network
        time.sleep(2.0)
        logging.info("✓ Marker stream stable and ready")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        raise

    #start data collector
    stop_event = threading.Event()
    collector = LSLDataCollector(
        stop_event, neurofeedback_processor, marker_source_id=marker_source_id
    )
    collector.start()
    
    # Wait for collector to be ready (streams resolved)
    if not collector.ready_event.wait(timeout=15):
        raise RuntimeError("Timed out waiting for LSL collector to resolve streams")
    if collector.startup_error is not None:
        raise RuntimeError("LSL collector failed to start") from collector.startup_error
    logging.info("✓ Data collector ready and connected")

   # run experiment
    run_neurofeedback_experiment(neurofeedback_processor, marker_outlet)


    logging.info("Experiment complete. Stopping data collector thread...")
    stop_event.set()
    collector.join(timeout=2.0)  # Wait max 2 seconds for clean shutdown

    logging.info("All processes complete. Exiting script.")





    
    #try:
        # Start the data collector thread
       # logging.info("Starting LSL data collector thread...")
      #  collector.start()
       # logging.info("✓ Data collector started")

        
        
        # Wait until the collector has resolved both EEG and marker streams.
        #if not collector.ready_event.wait(timeout=15):
            #raise RuntimeError("Timed out waiting for the LSL collector to become ready")
       # if collector.startup_error is not None:
           # raise RuntimeError("LSL collector failed to start") from collector.startup_error
        
        # Run using the same marker outlet created above.
        #logging.info("Launching neurofeedback experiment...")
        #run_neurofeedback_experiment(neurofeedback_processor, marker_outlet)
       # logging.info("✓ Experiment completed successfully")
        
    # except KeyboardInterrupt:
       # logging.info("KeyboardInterrupt received during experiment. Stopping...")
       # stop_event.set()
    #exit_code = 130
    #except Exception as e:
       # logging.exception("Experiment error - details above:")
       # stop_event.set()
       # exit_code = 1
    #finally:
        # ensure collector stops and join with a timeout
        #logging.info("Shutting down data collector...")
        #stop_event.set()
        #collector.join(timeout=5.0)
        #if collector.is_alive():
            #logging.warning("Collector thread failed to stop within 5s - may lose final data")
        
        # Convert to BIDS if we have an output
        #logging.info(f"Attempting BIDS conversion for {collector.output_csv}...")
        #try:
         #   convert_to_bids(collector.output_csv)
       # except Exception:
        #    logging.exception("BIDS conversion failed")
        
        #logging.info("="*70)
        #logging.info("Motor imagery experiment finished with LSL data.")
        #logging.info(f"Data saved to: {collector.output_csv}")
       # logging.info("="*70)

    #if exit_code != 0:
        #raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
