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
from scipy.signal import butter, filtfilt
import sys
sys.path.insert(0, '.')



# Import from current directory
from calculate_band_power import calculate_band_power
from fft_power_update import fft_power_update
from bandpass_filter import bandpass_filter
#import neurofeedback processor
from NeurofeedbackProcessor import NeurofeedbackProcessor
neurofeedback_processor= NeurofeedbackProcessor()


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



# Import lsl data collector class
from LSLDataCollector import LSLDataCollector







# import run_neurofeedback_experiment
from run_neurofeedback_experiment import run_neurofeedback_experiment








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
    missing_cols = [ch for ch in EEG_CHANNELS if ch not in df.columns]
    if missing_cols:
        logging.error(f"CSV is missing expected EEG channels: {missing_cols}")
        return

    # Extract EEG data
    data_eeg = []
    for ch in EEG_CHANNELS:
        col_data = pd.to_numeric(df[ch], errors="coerce").fillna(0).values
        data_eeg.append(col_data)
    data_eeg = np.array(data_eeg)  # shape: (n_channels, n_samples)

    # Process markers
    raw_markers = df["marker"].astype(str).replace({"nan": "", "None": ""}).values
    markers = []
    for i, m in enumerate(raw_markers):
        if m != "":
            markers.append((i, m))

    from mne import create_info
    from mne.io import RawArray
    info = create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLE_RATE, ch_types="eeg")
    raw = RawArray(data_eeg, info)

    # Create annotations from markers
    annotations_onset = []
    annotations_desc  = []
    for (idx, marker_val) in markers:
        onset_sec = idx / SAMPLE_RATE  # Alternatively, use LSL timestamps if available
        annotations_onset.append(onset_sec)
        annotations_desc.append(f"Marker/{marker_val}")

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
    stop_event = threading.Event()
    collector = LSLDataCollector(stop_event, neurofeedback_processor)
    collector.start()

    try:
        # run with real marker outlet (collector will resolve streams)
        run_neurofeedback_experiment(neurofeedback_processor)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received during experiment. Stopping...")
        stop_event.set()
    except Exception:
        logging.exception("Experiment error:")
        stop_event.set()
    finally:
        # ensure collector stops and join with a timeout
        stop_event.set()
        collector.join(timeout=5.0)
        if collector.is_alive():
            logging.warning("Collector thread failed to stop within 5s")
        # Convert to BIDS if we have an output
        try:
            convert_to_bids(collector.output_csv)
        except Exception:
            logging.exception("BIDS conversion failed")
        logging.info("Motor imagery experiment finished with LSL data.")

if __name__ == "__main__":
    main()
