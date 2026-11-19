import threading
import csv
import logging
import os
import time
import sys
from datetime import datetime
from collections import deque
from pylsl import resolve_byprop, StreamInlet

from config import (
    EEG_STREAM_NAME,
    MARKER_STREAM_NAME,
    EEG_CHANNELS,
    CSV_EEG_CHANNEL_INDICES,
    CSV_EEG_CHANNELS,
    MERGE_THRESHOLD,
    POLL_SLEEP,
    RAW_DATA_DIR,
    MARKER_RIGHT,
    MARKER_LEFT,
    MARKER_FEET,
    MARKER_NO_MOVEMENT,
    MARKER_STOP

)


class LSLDataCollector(threading.Thread):
    """
    Background thread that collects and synchronizes EEG and marker data.
    """
    def __init__(self, stop_event, neurofeedback_processor, marker_source_id=None):
        super().__init__()
        self.daemon = True  # Make thread a daemon so it exits when main program does
        self.stop_event = stop_event
        self.neurofeedback_processor = neurofeedback_processor
        self.marker_source_id = marker_source_id
        self.ready_event = threading.Event()
        self.startup_error = None
        self.eeg_buffer = deque()
        self.marker_buffer = deque()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure data directory exists before creating files
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        self.output_csv = os.path.join(RAW_DATA_DIR, f"MI_EEG_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0
        
        # Create a separate buffer for high-priority neurofeedback processing
        self.neurofeedback_queue = deque(maxlen=5)  # Small buffer to prevent overflow

    def resolve_streams(self):
        """Resolve EEG and Marker LSL streams with detailed logging."""
        try:
            logging.info(f"Resolving EEG LSL stream (name='{EEG_STREAM_NAME}', timeout=10s)...")
            from pylsl import cf_string, resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                error_msg = (f"No EEG stream found with name '{EEG_STREAM_NAME}' after 10s timeout. "
                           f"Verify that the EEG data source (BrainVision/synthetic stream) is running.")
                logging.error(error_msg)
                self.startup_error = RuntimeError(error_msg)
                self.ready_event.set()  # Signal main thread even on error
                return
            
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"✓ EEG stream resolved. Clock offset: {self.clock_offset:.6f}s")

            logging.info(f"Resolving Marker LSL stream (name='{MARKER_STREAM_NAME}', timeout=10s)...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                error_msg = (f"No Marker stream found with name '{MARKER_STREAM_NAME}' after 10s timeout. "
                           f"Verify that the marker stream outlet is still active in the main process.")
                logging.error(error_msg)
                self.startup_error = RuntimeError(error_msg)
                self.ready_event.set()  # Signal main thread even on error
                return

            marker_stream = next(
                (
                    stream for stream in marker_streams
                    if stream.channel_format() == cf_string
                    and stream.channel_count() == 1
                    and (
                        self.marker_source_id is None
                        or stream.source_id() == self.marker_source_id
                    )
                ),
                None,
            )
            if marker_stream is None:
                available_streams = [
                    {
                        "source_id": stream.source_id(),
                        "channel_format": stream.channel_format(),
                        "channel_count": stream.channel_count(),
                    }
                    for stream in marker_streams
                ]
                error_msg = (
                    f"No matching string marker stream found for '{MARKER_STREAM_NAME}'. "
                    f"Expected source_id={self.marker_source_id!r}, one string channel; "
                    f"found {available_streams!r}."
                )
                logging.error(error_msg)
                self.startup_error = RuntimeError(error_msg)
                self.ready_event.set()
                return

            self.marker_inlet = StreamInlet(marker_stream, max_buflen=360)

            logging.info(
                "Marker stream resolved: source_id=%s format=string channels=1",
                marker_stream.source_id(),
            )
            logging.info("✓ LSL streams ready. Starting data collection...")
            self.ready_event.set()  # Signal that we're ready
        except Exception as e:
            logging.exception("Failed to resolve LSL streams:")
            self.startup_error = e
            self.ready_event.set()  # Signal main thread even on error
       

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        # Define valid trial markers (not including STOP)
        valid_trial_markers = {MARKER_RIGHT, MARKER_LEFT, MARKER_FEET, MARKER_NO_MOVEMENT}
        
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
            
                    row = [ts_eeg] + [eeg_data[i] for i in CSV_EEG_CHANNEL_INDICES] + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
        
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(CSV_EEG_CHANNELS)) + [marker]
                writer.writerow(row)

    def run(self):
        from pylsl import resolve_byprop, StreamInlet
        self.resolve_streams()
        if self.startup_error is not None:
            logging.error(
                "Data collector could not connect to its LSL streams; "
                f"no CSV was created. Expected output path: {self.output_csv}"
            )
            return
    
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + CSV_EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self.stop_event.is_set():
                    try:
                        # Use chunk processing for efficiency when available
                        samples, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=32)
                        if samples and timestamps:
                            for sample, ts in zip(samples, timestamps):
                                # Process for neurofeedback immediately
                                self.neurofeedback_processor.process_sample(sample)
                                
                                # Buffer for data saving
                                self.eeg_buffer.append((ts, sample))
                    except Exception as e:
                        # If chunk not available, fallback to single sample
                        try:
                            sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                            if sample_eeg is not None and ts_eeg is not None:
                                self.eeg_buffer.append((ts_eeg, sample_eeg))
                                # Process sample for neurofeedback
                                self.neurofeedback_processor.process_sample(sample_eeg)
                        except Exception as inner_e:
                            logging.exception("Error pulling EEG sample:")
                    
                    try:
                        sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                        if sample_marker is not None and ts_marker is not None:
                            adjusted_ts_marker = ts_marker - self.clock_offset
                            marker_val = sample_marker[0] 
                            valid_markers = {
                                MARKER_RIGHT,
                                MARKER_LEFT,
                                MARKER_FEET,
                                MARKER_NO_MOVEMENT,
                                MARKER_STOP,
                            }
                            if marker_val in valid_markers:
                                self.marker_buffer.append((adjusted_ts_marker, marker_val))
                            else:
                                logging.warning(f"Ignoring unknown marker value: {marker_val!r}")
                    except Exception as e:
                        pass

                    # Merge data from both buffers (file saving is lower priority than processing)
                    while self.eeg_buffer and self.marker_buffer:
                        ts_eeg, eeg_data = self.eeg_buffer[0]
                        ts_marker, marker = self.marker_buffer[0]
                        if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                            row = [ts_eeg] + [eeg_data[i] for i in CSV_EEG_CHANNEL_INDICES] + [marker]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                            self.marker_buffer.popleft()
                        elif ts_marker < ts_eeg:
                            row = [ts_marker] + ([""] * len(CSV_EEG_CHANNELS)) + [marker]
                            writer.writerow(row)
                            self.marker_buffer.popleft()
                        else:
                            row = [ts_eeg] + [eeg_data[i] for i in CSV_EEG_CHANNEL_INDICES] + [""]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                    
                    # Flush buffer periodically to avoid excessive memory use
                    if len(self.eeg_buffer) > 1000:
                        while len(self.eeg_buffer) > 500:
                            ts_eeg, eeg_data = self.eeg_buffer.popleft()
                            row = [ts_eeg] + [eeg_data[i] for i in CSV_EEG_CHANNEL_INDICES] + [""]
                            writer.writerow(row)
                            
                    time.sleep(POLL_SLEEP)
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. CSV path: {self.output_csv}")
