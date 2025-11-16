import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd

def extract_acoustic_features(sound):
    """Extract various acoustic features from a sound object"""
    # Pitch features
    pitch = sound.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    
    # Formant features
    formants = sound.to_formant_burg()
    num_frames = call(formants, "Get number of frames")
    
    # Get mean F1, F2, F3
    f1_values = []
    f2_values = []
    f3_values = []
    
    for frame in range(1, num_frames + 1):
        f1 = call(formants, "Get value at time", 1, formants.t_grid()[frame - 1], "Hertz", "Linear")
        f2 = call(formants, "Get value at time", 2, formants.t_grid()[frame - 1], "Hertz", "Linear")
        f3 = call(formants, "Get value at time", 3, formants.t_grid()[frame - 1], "Hertz", "Linear")
        
        if not np.isnan(f1) and f1 > 0:
            f1_values.append(f1)
        if not np.isnan(f2) and f2 > 0:
            f2_values.append(f2)
        if not np.isnan(f3) and f3 > 0:
            f3_values.append(f3)
    
    mean_f1 = np.mean(f1_values) if f1_values else np.nan
    mean_f2 = np.mean(f2_values) if f2_values else np.nan
    mean_f3 = np.mean(f3_values) if f3_values else np.nan
    
    # Intensity
    intensity = sound.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
    
    # Voice quality measures - Jitter and Shimmer
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Harmonicity
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = call(harmonicity, "Get mean", 0, 0)
    
    # Collect all features in a dictionary
    features = {
        "Mean Pitch (Hz)": mean_pitch,
        "Min Pitch (Hz)": min_pitch,
        "Max Pitch (Hz)": max_pitch,
        "Mean F1 (Hz)": mean_f1,
        "Mean F2 (Hz)": mean_f2,
        "Mean F3 (Hz)": mean_f3,
        "Mean Intensity (dB)": mean_intensity,
        "Min Intensity (dB)": min_intensity,
        "Max Intensity (dB)": max_intensity,
        "Jitter (local %)": jitter_local * 100,  # Convert to percentage
        "Shimmer (local %)": shimmer_local * 100,  # Convert to percentage
        "Mean HNR (dB)": mean_hnr
    }
    
    return features

# Example usage
sound = parselmouth.Sound("uploads/cMajor-1-octave.wav")
features = extract_acoustic_features(sound)

# Print results
for feature, value in features.items():
    print(f"{feature}: {value:.4f}")

# Or convert to DataFrame
df = pd.DataFrame([features])
print(df)