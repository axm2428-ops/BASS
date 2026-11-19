import logging
import random
import sys
import time
import numpy as np
from psychopy import visual, core, event

from config import (

    BASELINE_END,
    BASELINE_START,
    EEG_STREAM_NAME,
    MARKER_FEET,
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
    MARKER_STOP,
    MARKER_NO_MOVEMENT,
    RAW_DATA_DIR,
    FEEDBACK_BAR_MAX_WIDTH,
    DISPLAY_UPDATE_INTERVAL,
    INITIAL_BASELINE_DURATION
)


def run_neurofeedback_experiment(neurofeedback_processor, marker_outlet):
    """
    Run the neurofeedback experiment with visualizations using PsychoPy.
    
    Parameters:
    
    -----------
    neurofeedback_processor : NeurofeedbackProcessor
        The processor instance that handles EEG signal processing
    marker_outlet : pylsl.StreamOutlet
        The LSL marker outlet for sending experiment markers (pre-created by main script)
    """
    logging.info("\n" + "="*70)
    logging.info("NEUROFEEDBACK EXPERIMENT STARTING")
    logging.info(f"Neurofeedback Processor: {type(neurofeedback_processor).__name__}")
    logging.info("✓ Using pre-created marker outlet")

    # PsychoPy setup
    try:
        logging.info("Initializing PsychoPy window...")
        from psychopy import visual, core, event, clock
        
        # Set up monitor for optimal performance
        win = visual.Window(
            size=(1024, 768),
            color="black",
            units="norm",
            fullscr=False,
            allowGUI=False,  # Disable GUI for better performance
            waitBlanking=False  # Disable waiting for screen refresh for lower latency
        )
        win.recordFrameIntervals = True  # Track frame timing
        
        logging.info("✓ PsychoPy window created successfully (1024x768, windowed mode)")
    except Exception as e:
        logging.exception("ERROR: Failed to initialize PsychoPy window. Is a display connected?")
        raise

    # Create visual stimuli
    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")
    countdown_text = visual.TextStim(win, text="", height=0.2, color="white", pos=(0, 0.2))
    
    # Create channel indicator
    channel_text = visual.TextStim(win, text="", height=0.05, color="white",
                               pos=(0, -0.7))
    
    # Create Mu band feedback meter (right side) - similar to train_response.py
    meter_width = 0.2
    meter_height = 0.6
    mu_meter_pos = (0.7, 0)  # Position on right side of screen
    mu_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=mu_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    mu_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=mu_meter_pos, 
        fillColor='green',
        autoLog=False
    )
    mu_meter_text = visual.TextStim(
        win=win, 
        text="Mu Band Power", 
        height=0.05,
        pos=(mu_meter_pos[0], mu_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create Beta band feedback meter (left side)
    beta_meter_pos = (-0.7, 0)  # Position on left side of screen
    beta_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=beta_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    beta_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=beta_meter_pos, 
        fillColor='blue',
        autoLog=False
    )
    beta_meter_text = visual.TextStim(
        win=win, 
        text="Beta Band Power", 
        height=0.05,
        pos=(beta_meter_pos[0], beta_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create central feedback bar with optimized settings (keep this from original)
    feedback_bar_bg = visual.Rect(
        win=win,
        width=FEEDBACK_BAR_MAX_WIDTH,
        height=0.2,
        fillColor="gray",
        lineColor="white",
        pos=(0, -0.5),
        autoLog=False
    )
    
    feedback_bar = visual.Rect(
        win=win,
        width=0,  # Will be updated based on ERD
        height=0.15,
        fillColor="green",
        lineColor=None,
        pos=(-FEEDBACK_BAR_MAX_WIDTH/2, -0.5),  # Centered position
        anchor="left",  # Anchor to left for width updates
        autoLog=False
    )

    # Generate randomized trial list with both hands
    import math
    base_trials = ["right_arm", "left_arm", "feet"] * (NUM_TRIALS // 3)
    
    # Add no-movement trials based on probability
    final_trials = []
    for trial in base_trials:
        if random.random() < NO_MOVEMENT_PROBABILITY:
            final_trials.append("no_movement")
        else:
            final_trials.append(trial)
            
    # Shuffle all trials
    random.shuffle(final_trials)
    
    # Show initial instructions
    logging.info("Displaying initial instructions...")
    instruction_text.text = "Motor Imagery Neurofeedback\n\nImagine moving your hand when instructed\n\nDo not move when 'NO MOVEMENT' is shown\n\nThe feedback bars will show your brain activity\n\nPress SPACE to begin"
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    logging.info("User confirmed - starting baseline collection")
    
    # Initial 10-second baseline collection
    instruction_text.text = "Collecting baseline\n\nPlease relax and remain still"
    neurofeedback_processor.start_initial_baseline_collection()
    logging.info(f"Starting initial baseline collection ({INITIAL_BASELINE_DURATION}s)...")
    
    baseline_clock = core.Clock()
    baseline_clock.reset()
    
    # Countdown for initial baseline
    while baseline_clock.getTime() < INITIAL_BASELINE_DURATION:
        remaining = int(INITIAL_BASELINE_DURATION - baseline_clock.getTime())
        instruction_text.draw()
        countdown_text.text = f"{remaining}s"
        win.flip()
        countdown_text.draw()
        win.flip()
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            win.close()
            neurofeedback_processor.stop_initial_baseline_collection()
            logging.info("User cancelled during baseline")
            return
    
    neurofeedback_processor.stop_initial_baseline_collection()
    logging.info("✓ Initial baseline collection completed")
    
    # Indicate baseline completion
    instruction_text.text = "Baseline Completed\n\nThe experiment will begin shortly"
    instruction_text.draw()
    win.flip()
    core.wait(2.0)
    logging.info(f"Beginning {len(final_trials)} trials...")

    # Define channel display text mapping (reused for all trials)
    channel_display = {
        "right_arm": "Using CH3 (Contralateral Hemisphere)",
        "left_arm": "Using CH6 (Contralateral Hemisphere)",
        "feet": "Using CH3 and CH6 (Contralateral Hemisphere)",
        "no_movement": "Maintain resting state (no movement)"
    }
    
    # Define instruction display mapping (reused for all trials)
    instruction_display = {
        "right_arm": "RIGHT_ARM LIMB",
        "left_arm": "LEFT_ARM LIMB",
        "feet": "FEET LIMB",
        "no_movement": "NO MOVEMENT"
    }
    
    # Define marker mapping (reused for all trials)
    marker_map = {
        "right_arm": MARKER_RIGHT,
        "left_arm": MARKER_LEFT,
        "feet": MARKER_FEET,
        "no_movement": MARKER_NO_MOVEMENT
    }

    # Main experiment loop
    for trial_num, limb in enumerate(final_trials, 1):
        # Set the active limb for this trial (handles regular limbs and no-movement)
        neurofeedback_processor.set_active_limb(limb)
        
        # Update the channel indicator display
        channel_text.text = channel_display[limb]
        # Display get ready message
        ready_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(1.0)
        
        # Show instruction (right limb, left limb, or no movement)
        instruction_text.text = instruction_display[limb]
        
        instruction_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)
        
        # Start collecting baseline data 3 seconds before the START cue
        # We use a separate clock to time the baseline collection
        baseline_clock = core.Clock()
        baseline_clock.reset()
        neurofeedback_processor.start_baseline_collection()
        
        # Wait for baseline collection (-3s to -1s before START)
        # Continue showing instruction during this time
        while baseline_clock.getTime() < abs(BASELINE_START - BASELINE_END):  # 2 seconds
            instruction_text.draw()
            channel_text.draw()
            win.flip()
            
            # Check for quit
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop baseline collection and prepare for processing
        neurofeedback_processor.stop_baseline_collection()
        
        # Show START cue and send start marker
        instruction_text.text = "START"
        instruction_text.draw()
        channel_text.draw()
        # Schedule marker to be sent on next flip
        marker_val = marker_map[limb]
        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        # Delay to allow LSL to transmit marker (50ms ensures network transport)
        time.sleep(0.05)
        logging.info(f"Sent START marker: {marker_val} for {limb}")
        
        # Start real-time processing after the START cue
        neurofeedback_processor.start_processing()
        
        # Initialize feedback timer and display timer
        feedback_timer = core.Clock()
        display_timer = core.Clock()
        feedback_timer.reset()
        display_timer.reset()
        
        # Continuous feedback during imagery period
        while feedback_timer.getTime() < IMAGERY_DURATION:
            # Only update display at the target refresh rate
            if display_timer.getTime() >= DISPLAY_UPDATE_INTERVAL:
                # Reset display timer
                display_timer.reset()
                
                # Get current ERD values with smoothing for visual appeal
                smoothed_mu = neurofeedback_processor.get_smoothed_erd_mu()
                smoothed_beta = neurofeedback_processor.get_smoothed_erd_beta()
                
                # Update central feedback bar width based on mu ERD value
                bar_width = min(smoothed_mu * FEEDBACK_BAR_MAX_WIDTH, FEEDBACK_BAR_MAX_WIDTH)
                feedback_bar.width = bar_width
                
                # Update mu meter height
                mu_meter_fill.height = smoothed_mu * meter_height
                mu_meter_fill.pos = (mu_meter_pos[0], mu_meter_pos[1] - (meter_height/2) + (smoothed_mu * meter_height/2))
                
                # Update beta meter height
                beta_meter_fill.height = smoothed_beta * meter_height
                beta_meter_fill.pos = (beta_meter_pos[0], beta_meter_pos[1] - (meter_height/2) + (smoothed_beta * meter_height/2))
                
                # Update central bar color based on mu ERD value
                if smoothed_mu < 0.2:
                    feedback_bar.fillColor = "red"
                elif smoothed_mu < 0.6:
                    feedback_bar.fillColor = "yellow"
                else:
                    feedback_bar.fillColor = "green"
                
                # Draw feedback
                instruction_text.text = "HOLD"
                instruction_text.draw()
                
                # Draw mu and beta meters
                mu_meter_bg.draw()
                mu_meter_fill.draw()
                mu_meter_text.draw()
                beta_meter_bg.draw()
                beta_meter_fill.draw()
                beta_meter_text.draw()
                
                # Draw central feedback bar
                feedback_bar_bg.draw()
                feedback_bar.draw()
                
                # Draw channel indicator
                channel_text.draw()
                
                win.flip()
            
            # Process any events to prevent blocking
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop real-time processing
        neurofeedback_processor.stop_processing()
        
        # Show STOP and send stop marker
        instruction_text.text = "STOP"
        instruction_text.draw()
        channel_text.draw()
        win.callOnFlip(lambda s=MARKER_STOP: marker_outlet.push_sample([s]))
        win.flip()
        # Delay to allow LSL to transmit marker (50ms ensures network transport)
        time.sleep(0.05)
        logging.info(f"Sent STOP marker")
        core.wait(1.0)
        
        # Inter-trial interval
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
        
        # Report frame timing stats
        if win.recordFrameIntervals:
            frame_times = win.frameIntervals
            if frame_times:
                mean_frame_time = np.mean(frame_times)
                std_frame_time = np.std(frame_times)
                logging.info(f"Frame timing: Mean={mean_frame_time*1000:.1f}ms, Std={std_frame_time*1000:.1f}ms")
                win.frameIntervals = []  # Reset for next trial
        
        # Log completion with appropriate description
        if limb == "no_movement":
            logging.info(f"Completed trial {trial_num}/{len(final_trials)}: no movement")
        else:
            logging.info(f"Completed trial {trial_num}/{len(final_trials)}: {limb} limb")
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            break

    # Cleanup
    win.close()
    logging.info("="*70)
    logging.info("✓ Neurofeedback experiment finished successfully")
    logging.info("="*70)
