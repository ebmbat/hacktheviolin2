#!/usr/bin/env python3
"""
Pitch Analyzer - A CLI tool to analyze pitch frequencies in audio files.

This tool uses the Parselmouth library (Python interface to Praat) to extract
pitch information from audio files and provide detailed analysis.
"""

import argparse
import os
import sys
import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
from tabulate import tabulate

# Standard range of a 4-string violin: G3 to E7
VIOLIN_MIN_PITCH=196.0
VIOLIN_MAX_PITCH=2637.02

# List of note-frequency tuples
notes_and_frequencies = [
    ('G3', 196.0), ('G#3', 207.65), ('A3', 220.0), ('A#3', 233.08), ('B3', 246.94),
    ('C4', 261.63), ('C#4', 277.18), ('D4', 293.66), ('D#4', 311.13), ('E4', 329.63),
    ('F4', 349.23), ('F#4', 369.99), ('G4', 392.0), ('G#4', 415.3), ('A4', 440.0),
    ('A#4', 466.16), ('B4', 493.88), ('C5', 523.25), ('C#5', 554.37), ('D5', 587.33),
    ('D#5', 622.25), ('E5', 659.26), ('F5', 698.46), ('F#5', 739.99), ('G5', 783.99),
    ('G#5', 830.61), ('A5', 880.0), ('A#5', 932.33), ('B5', 987.77), ('C6', 1046.5),
    ('C#6', 1108.73), ('D6', 1174.66), ('D#6', 1244.51), ('E6', 1318.51), ('F6', 1396.91),
    ('F#6', 1479.98), ('G6', 1567.98), ('G#6', 1661.22), ('A6', 1760.0), ('A#6', 1864.66),
    ('B6', 1975.53), ('C7', 2093.0), ('C#7', 2217.46), ('D7', 2349.32), ('D#7', 2489.02),
    ('E7', 2637.02)
]

# Creating a dictionary from the list of tuples
notes_to_frequencies = dict(notes_and_frequencies)

# Sharp-to-flat mappings
sharp_to_flat_notes = [
    ('G#3', 'A♭3'), ('A#3', 'B♭3'), ('C#4', 'D♭4'), ('D#4', 'E♭4'),
    ('F#4', 'G♭4'), ('G#4', 'A♭4'), ('A#4', 'B♭4'), ('C#5', 'D♭5'),
    ('D#5', 'E♭5'), ('F#5', 'G♭5'), ('G#5', 'A♭5'), ('A#5', 'B♭5'),
    ('C#6', 'D♭6'), ('D#6', 'E♭6'), ('F#6', 'G♭6'), ('G#6', 'A♭6'),
    ('A#6', 'B♭6'), ('C#7', 'D♭7')
]

# Scales mappings
one_octave_major_scales = {
    'G major': ('G3', 'A3', 'B3', 'C4', 'D4', 'E4', 'F#4', 'G4'),
    'A major': ('A3', 'B3', 'C#4', 'D4', 'E4', 'F#4', 'G#4', 'A4'),
    'B major': ('B3', 'C#4', 'D#4', 'E4', 'F#4', 'G#4', 'A#4', 'B4'),
    'C major': ('C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'),
    'D major': ('D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5'),
    'E major': ('E4', 'F#4', 'G#4', 'A4', 'B4', 'C#5', 'D#5', 'E5'),
    'F major': ('F4', 'G4', 'A4', 'B♭4', 'C5', 'D5', 'E5', 'F5')
}

one_octave_natural_minor_scales = {
    'G minor': ('G3', 'A3', 'B♭3', 'C4', 'D4', 'E♭4', 'F#4', 'G4'),
    'A minor': ('A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4'),
    'B minor': ('B3', 'C#4', 'D4', 'E4', 'F#4', 'G4', 'A4', 'B4'),
    'C minor': ('C4', 'D4', 'E♭4', 'F4', 'G4', 'A♭4', 'B♭4', 'C5'),
    'D minor': ('D4', 'E4', 'F4', 'G4', 'A4', 'B♭4', 'C4', 'D5'),
    'E minor': ('E4', 'F#4', 'G4', 'A4', 'B4', 'C4', 'D4', 'E5'),
    'F minor': ('F4', 'G4', 'A♭4', 'B♭4', 'C5', 'D4', 'E4', 'F5')
}

one_octave_harmonic_minor_scales = {
    'G minor': ('G3', 'A3', 'B♭3', 'C4', 'D4', 'E4', 'F#4', 'G4'),
    'A minor': ('A3', 'B3', 'C4', 'D4', 'E4', 'F#4', 'G#4', 'A4'),
    'B minor': ('B3', 'C#4', 'D4', 'E4', 'F#4', 'G#4', 'A#4', 'B4'),
    'C minor': ('C4', 'D4', 'E♭4', 'F4', 'G4', 'A♭4', 'B♭4', 'C5'),
    'D minor': ('D4', 'E4', 'F4', 'G4', 'A4', 'B♭4', 'C#5', 'D5'),
    'E minor': ('E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D#5', 'E5'),
    'F minor': ('F4', 'G4', 'A♭4', 'B♭4', 'C5', 'D5', 'E6', 'F5')
}

def extract_pitch(sound, min_pitch=VIOLIN_MIN_PITCH, max_pitch=VIOLIN_MAX_PITCH, time_step=0.0):
    # 2025-05-10: removed the time_stamp optional argument - mpz
    """
    Extract pitch information from a sound object.
    Args:
        sound: Parselmouth Sound object
        min_pitch: Minimum pitch in Hz (Violin lowest pitch: 196.00)
        max_pitch: Maximum pitch in Hz (Violin highest pitch: 2637.02)
        time_step: Time step in seconds (default: 0.0, which uses Praat's default)
        
    Returns:
        Parselmouth Pitch object
    """
    return sound.to_pitch(pitch_floor=min_pitch, pitch_ceiling=max_pitch, time_step=0.05)


def analyze_pitch(pitch):
    """
    Analyze pitch data and return statistics.
    Args:
        pitch: Parselmouth Pitch object
    Returns:
        Dictionary containing pitch statistics
    """
    # Extract pitch values (excluding unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]
    
    # Calculate statistics
    if len(pitch_values) > 0:
        stats = {
            'mean': np.mean(pitch_values),
            'median': np.median(pitch_values),
            'std': np.std(pitch_values),
            'min': np.min(pitch_values),
            'max': np.max(pitch_values),
            'range': np.max(pitch_values) - np.min(pitch_values),
            'q1': np.percentile(pitch_values, 25),
            'q3': np.percentile(pitch_values, 75)
        }
    else:
        stats = {
            'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'range': 0,
            'q1': 0, 'q3': 0
        }
    
    return stats


def generate_pitch_plot(sound, pitch, note_transitions=None, output_path=None):
    """
    Generate a plot of the pitch contour with detected notes.
    
    Args:
        sound: Parselmouth Sound object
        pitch: Parselmouth Pitch object
        note_transitions: List of note transitions (start_time, end_time, freq, note)
        output_path: Path to save the plot (if None, display plot)
        
    Returns:
        None
    """
    # Create figure with subplots
    n_plots = 3 if note_transitions else 2
    plt.figure(figsize=(14, 10))
    
    # Plot waveform
    plt.subplot(n_plots, 1, 1)
    plt.plot(np.linspace(0, sound.duration, len(sound.values[0])), sound.values[0])
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Extract pitch data
    pitch_values = pitch.selected_array['frequency']
    pitch_times = pitch.xs()
    
    # Plot pitch
    plt.subplot(n_plots, 1, 2)
    pitch_values[pitch_values==0] = np.nan  # Replace unvoiced samples with NaN
    plt.plot(pitch_times, pitch_values, 'o', markersize=2)
    plt.grid(True)
    plt.title('Pitch Contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, pitch.ceiling)
    
    # Plot note transitions if available
    if note_transitions:
        plt.subplot(n_plots, 1, 3)
        
        # Create a colormap with enough colors
        cmap = plt.cm.get_cmap('hsv', 12)
        
        # Plot each note segment as a colored rectangle
        for start_time, end_time, freq, note in note_transitions:
            # Extract note without octave for color mapping
            note_name = note.rstrip('0123456789')
            # Map note to color (C=0, C#=1, etc.)
            note_idx = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(note_name)
            color = cmap(note_idx / 12)
            
            plt.axvspan(start_time, end_time, alpha=0.3, color=color)
            # Add note label at the center of the segment
            plt.text((start_time + end_time) / 2, 0.5, note, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=8, fontweight='bold')
        
        plt.grid(False)
        plt.title('Detected Notes')
        plt.xlabel('Time (s)')
        plt.yticks([])  # Hide Y axis
        plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def find_note(frequency):
    """
    Convert a frequency to the closest musical note name.
    Args:
        frequency: Frequency in Hz
    Returns:
        String containing note name and octave
    """
    if frequency <= 0:
        return "N/A"
        
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # A4 is 440 Hz, which is midi note 69
    a4_freq = 440.0
    a4_midi = 69
    
    # Convert frequency to midi note number
    midi_note = round(12 * np.log2(frequency / a4_freq) + a4_midi)
    
    # Calculate note name and octave
    note_name = note_names[midi_note % 12]
    octave = (midi_note - 12) // 12
    
    return f"{note_name}{octave}"


def get_predominant_frequencies(pitch, n=5):
    """
    Find the n most predominant frequencies in the audio file.
    Args:
        pitch: Parselmouth Pitch object
        n: Number of frequencies to return
    Returns:
        List of tuples (frequency, count, percentage, note)
    """
    # Extract valid pitch values (non-zero)
    pitch_values = pitch.selected_array['frequency']
    valid_pitches = pitch_values[pitch_values != 0]
    
    if len(valid_pitches) == 0:
        return []
    
    # Round to the nearest Hz for binning
    rounded_pitches = np.round(valid_pitches).astype(int)
    
    # Count occurrences
    unique, counts = np.unique(rounded_pitches, return_counts=True)
    
    # Sort by count (descending)
    sorted_indices = np.argsort(-counts)
    
    # Get the top n frequencies
    top_frequencies = []
    for i in range(min(n, len(unique))):
        idx = sorted_indices[i]
        freq = unique[idx]
        count = counts[idx]
        percentage = (count / len(valid_pitches)) * 100
        note = find_note(freq)
        top_frequencies.append((freq, count, percentage, note))
    
    return top_frequencies


def detect_note_transitions(pitch, smooth_window=3, threshold_cents=50):
    """
    Detect transitions between different musical notes in the audio.
    Args:
        pitch: Parselmouth Pitch object
        smooth_window: Window size for smoothing pitch values (default: 3)
        threshold_cents: Minimum difference in cents to consider a note change (default: 50)
                         100 cents = 1 semitone
    Returns:
        List of tuples (start_time, end_time, frequency, note)
    """
    pitch_values = pitch.selected_array['frequency']
    times = pitch.xs()
    
    if len(times) == 0:
        return []
    
    # Smooth pitch values with a simple moving average
    if smooth_window > 1:
        smoothed_values = np.zeros_like(pitch_values)
        for i in range(len(pitch_values)):
            start = max(0, i - smooth_window // 2)
            end = min(len(pitch_values), i + smooth_window // 2 + 1)
            window_values = pitch_values[start:end]
            valid_values = window_values[window_values > 0]
            if len(valid_values) > 0:
                smoothed_values[i] = np.mean(valid_values)
            else:
                smoothed_values[i] = 0
    else:
        smoothed_values = pitch_values.copy()
    
    # Initialize note transitions
    note_transitions = []
    current_start = None
    current_freq = None
    min_duration = 0.05  # Minimum note duration in seconds
    
    # Helper function to convert frequency difference to cents
    def cents_difference(f1, f2):
        if f1 <= 0 or f2 <= 0:
            return 0
        return 1200 * np.log2(f2 / f1)
    
    # Detect note transitions
    for i in range(len(times)):
        freq = smoothed_values[i]
        
        if freq <= 0:  # Unvoiced frame
            if current_start is not None:
                # End current note if it was long enough
                if times[i-1] - times[current_start] >= min_duration:
                    avg_freq = np.mean(smoothed_values[current_start:i][smoothed_values[current_start:i] > 0])
                    note = find_note(avg_freq)
                    note_transitions.append((times[current_start], times[i-1], avg_freq, note))
                current_start = None
                current_freq = None
            continue
        
        # First voiced frame
        if current_start is None:
            current_start = i
            current_freq = freq
            continue
        
        # Check if we've moved to a new note
        if abs(cents_difference(current_freq, freq)) > threshold_cents:
            # End current note if it was long enough
            if times[i-1] - times[current_start] >= min_duration:
                avg_freq = np.mean(smoothed_values[current_start:i][smoothed_values[current_start:i] > 0])
                note = find_note(avg_freq)
                note_transitions.append((times[current_start], times[i-1], avg_freq, note))
            
            # Start new note
            current_start = i
            current_freq = freq
    
    # Add the final note segment if there is one
    if current_start is not None and current_start < len(times) - 1:
        i = len(times) - 1
        if times[i] - times[current_start] >= min_duration:
            avg_freq = np.mean(smoothed_values[current_start:][smoothed_values[current_start:] > 0])
            note = find_note(avg_freq)
            note_transitions.append((times[current_start], times[i], avg_freq, note))
    
    return note_transitions


def main():
    """Main function to process CLI arguments and analyze audio file."""
    parser = argparse.ArgumentParser(description='Analyze pitch information in audio files')
    
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('--min-pitch', type=float, default=VIOLIN_MIN_PITCH, 
                        help='Minimum pitch in Hz (default: G3, 196.00 Hz)')
    parser.add_argument('--max-pitch', type=float, default=VIOLIN_MAX_PITCH, 
                        help='Maximum pitch in Hz (default: E7, 2637.02 Hz)')
    parser.add_argument('--time-step', type=float, default=0.0, 
                        help='Time step in seconds (default: 0.0, which uses Praat\'s default)')
    parser.add_argument('--plot', action='store_true', 
                        help='Generate and display a pitch plot')
    parser.add_argument('--save-plot', type=str, 
                        help='Save the pitch plot to the specified file path')
    parser.add_argument('--simple', action='store_true', 
                        help='Output only the list of predominant frequencies')
    parser.add_argument('--top', type=int, default=5, 
                        help='Number of top frequencies to display (default: 5)')
    parser.add_argument('--detect-notes', action='store_true',
                        help='Detect and display note transitions in the audio')
    parser.add_argument('--smooth-window', type=int, default=3,
                        help='Window size for smoothing pitch values (default: 3)')
    parser.add_argument('--note-threshold', type=float, default=50.0,
                        help='Threshold in cents for detecting note changes (default: 50.0)')
    parser.add_argument('--export-notes', type=str,
                        help='Export detected notes to a CSV file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        return 1
    
    try:
        # Load sound file
        sound = parselmouth.Sound(args.input_file)
        
        # Extract pitch
        pitch = extract_pitch(sound, args.min_pitch, args.max_pitch, args.time_step)
        
        # Detect note transitions if requested
        note_transitions = None

        if args.detect_notes:
            note_transitions = detect_note_transitions(
                pitch, 
                smooth_window=args.smooth_window, 
                threshold_cents=args.note_threshold
            )
            
            # Export notes to CSV if requested
            if args.export_notes:
                import csv
                with open(args.export_notes, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Start Time (s)', 'End Time (s)', 'Duration (s)', 
                                    'Frequency (Hz)', 'Note', 'MIDI Note'])
                    
                    for start, end, freq, note in note_transitions:
                        # Calculate MIDI note number
                        if freq > 0:
                            a4_freq = 440.0
                            a4_midi = 69
                            midi_note = round(12 * np.log2(freq / a4_freq) + a4_midi)
                        else:
                            midi_note = ''
                            
                        writer.writerow([
                            f"{start:.3f}", 
                            f"{end:.3f}", 
                            f"{end - start:.3f}", 
                            f"{freq:.1f}", 
                            note, 
                            midi_note
                        ])
                print(f"Note transitions exported to {args.export_notes}")
        
        if args.simple:
            # Simple output format - only frequencies
            if args.detect_notes:
                # Output detected notes chronologically
                for start, end, freq, note in note_transitions:
                    print(f"{start:.3f}-{end:.3f}s: {note} ({freq:.1f} Hz, duration: {end-start:.3f}s)")
            else:
                # Output predominant frequencies
                predominant = get_predominant_frequencies(pitch, args.top)
                for freq, _, percentage, note in predominant:
                    print(f"{freq:.1f} Hz ({note}, {percentage:.1f}%)")
        else:
            # Detailed output
            print(f"\nAnalyzing: {os.path.basename(args.input_file)}")
            print(f"Duration: {sound.duration:.2f} seconds")
            print(f"Sample Rate: {sound.sampling_frequency} Hz")
            
            # Calculate and display statistics
            stats = analyze_pitch(pitch)
            print("\nPitch Statistics:")
            stat_table = [
                ["Mean", f"{stats['mean']:.1f} Hz", find_note(stats['mean'])],
                ["Median", f"{stats['median']:.1f} Hz", find_note(stats['median'])],
                ["Std Dev", f"{stats['std']:.1f} Hz", ""],
                ["Min", f"{stats['min']:.1f} Hz", find_note(stats['min'])],
                ["Max", f"{stats['max']:.1f} Hz", find_note(stats['max'])],
                ["Range", f"{stats['range']:.1f} Hz", ""],
                ["Q1 (25%)", f"{stats['q1']:.1f} Hz", find_note(stats['q1'])],
                ["Q3 (75%)", f"{stats['q3']:.1f} Hz", find_note(stats['q3'])]
            ]
            print(tabulate(stat_table, headers=["Measure", "Value", "Musical Note"]))
            
            # Display predominant frequencies
            predominant = get_predominant_frequencies(pitch, args.top)
            if predominant:
                print(f"\nTop {len(predominant)} Predominant Frequencies:")
                freq_table = [
                    [f"{freq:.1f} Hz", count, f"{percentage:.1f}%", note]
                    for freq, count, percentage, note in predominant
                ]
                print(tabulate(freq_table, headers=["Frequency", "Count", "Percentage", "Musical Note"]))
            else:
                print("\nNo valid pitch data found.")
            
            # Display detected note transitions
            if args.detect_notes and note_transitions:
                print(f"\nDetected {len(note_transitions)} Note Transitions:")
                notes_table = [
                    [f"{start:.3f}", f"{end:.3f}", f"{end-start:.3f}", f"{freq:.1f}", note]
                    for start, end, freq, note in note_transitions
                ]
                print(tabulate(notes_table, 
                              headers=["Start (s)", "End (s)", "Duration (s)", "Freq (Hz)", "Note"]))
        
        # Generate plot if requested
        if args.plot or args.save_plot:
            generate_pitch_plot(sound, pitch, note_transitions, args.save_plot)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())