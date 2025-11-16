import librosa
import numpy as np

def analyze_rhythm(audio_file, sr=None):
    """
    Analyzes the rhythm in a .wav audio file using librosa.

    Args:
        audio_file (str): Path to the .wav audio file.
        sr (int, optional): Sample rate. If None, librosa will use the file's native sample rate. Defaults to None.

    Returns:
        dict: A dictionary containing rhythm analysis results, including:
            - bpm (float): Estimated beats per minute.
            - beat_frames (np.ndarray): Frames corresponding to detected beats.
            - tempo_array (np.ndarray):  Tempo estimates at different time scales.
    """

    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=sr)

        # Estimate the tempo (beats per minute)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Get tempo estimates at different time scales using librosa.tempo
        tempo_array = librosa.tempo.beat_track(y=y, sr=sr)

        # Create a dictionary to store the results
        results = {
            'bpm': tempo,
            'beat_frames': beat_frames,
            'tempo_array': tempo_array  # Added for more detailed analysis
        }

        return results

    except Exception as e:
        print(f"Error analyzing rhythm: {e}")
        return None



def main():
    """
    Main function to demonstrate the rhythm analysis.  Prompts for file path and prints results.
    """

    audio_file = input("Enter the path to your .wav audio file: ")

    # Optional: Specify a sample rate.  If you don't, librosa will use the file's native rate.
    # sample_rate = 22050  # Example: Set to a specific sample rate

    results = analyze_rhythm(audio_file)
    # results = analyze_rhythm(audio_file, sr=sample_rate)  # Uncomment if you set a sample rate

    if results:
        print("\nRhythm Analysis Results:")
        print(f"Estimated BPM: {results['bpm']:.2f}")  # Format to 2 decimal places
        print(f"Beat Frames: {results['beat_frames']}")

        # Print some tempo estimates at different time scales
        print("\nTempo Estimates (at different time scales):")
        for i, tempo in enumerate(results['tempo_array']):
            print(f"Time Scale {i+1}: {tempo:.2f}")


if __name__ == "__main__":
    main()