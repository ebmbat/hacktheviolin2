import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveTimbreAnalyzer:
    """
    A comprehensive class to analyze all 13 MFCC coefficients and their relationship to tonal qualities of timbre
    """
    
    def __init__(self, n_mfcc=13, sr=22050, hop_length=512, n_fft=2048):
        """
        Initialize the ComprehensiveTimbreAnalyzer
        
        Parameters:
        - n_mfcc: Number of MFCC coefficients to extract (default: 13)
        - sr: Sample rate (default: 22050 Hz)
        - hop_length: Number of samples between successive frames (default: 512)
        - n_fft: Length of the FFT window (default: 2048)
        """
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Define comprehensive tonal qualities for each MFCC coefficient
        self.mfcc_tonal_qualities = {
            0: {
                'name': 'Overall Energy/Power',
                'tonal_quality': 'Loudness/Volume',
                'description': 'Total energy content - distinguishes loud vs soft sounds',
                'high_values': 'Loud, powerful sounds (brass, drums, forte dynamics)',
                'low_values': 'Soft, quiet sounds (whispers, distant sounds, piano dynamics)',
                'frequency_range': 'Full spectrum energy'
            },
            1: {
                'name': 'Spectral Centroid',
                'tonal_quality': 'Brightness/Darkness',
                'description': 'Center of mass of spectrum - primary brightness indicator',
                'high_values': 'Bright, sharp sounds (cymbals, high strings, flute)',
                'low_values': 'Dark, mellow sounds (bass, low brass, cello)',
                'frequency_range': 'Weighted frequency center'
            },
            2: {
                'name': 'Spectral Rolloff',
                'tonal_quality': 'Sharpness/Mellowness',
                'description': 'High-frequency content - affects perceived sharpness',
                'high_values': 'Sharp, cutting sounds (distorted guitar, sibilants)',
                'low_values': 'Smooth, mellow sounds (flute, filtered audio)',
                'frequency_range': 'High-frequency emphasis'
            },
            3: {
                'name': 'Spectral Slope',
                'tonal_quality': 'Harmonic Richness',
                'description': 'Rate of high-frequency decay - harmonic complexity',
                'high_values': 'Rich harmonics (violin, complex instruments)',
                'low_values': 'Simple tones (sine waves, pure tones)',
                'frequency_range': 'Harmonic structure'
            },
            4: {
                'name': 'Spectral Flux',
                'tonal_quality': 'Attack/Transient Character',
                'description': 'Rate of spectral change - attack characteristics',
                'high_values': 'Sharp attacks (piano, percussion, staccato)',
                'low_values': 'Smooth attacks (bowed strings, sustained pads)',
                'frequency_range': 'Temporal spectral changes'
            },
            5: {
                'name': 'Low-Mid Formant',
                'tonal_quality': 'Resonance/Body/Warmth',
                'description': 'Primary resonant peak - body and warmth',
                'high_values': 'Resonant, full-bodied, warm sounds',
                'low_values': 'Thin, lacking body, cold sounds',
                'frequency_range': '200-800 Hz resonance'
            },
            6: {
                'name': 'Mid Formant',
                'tonal_quality': 'Presence/Clarity',
                'description': 'Mid-frequency emphasis - vocal-like qualities',
                'high_values': 'Present, clear, forward, vocal-like',
                'low_values': 'Recessed, distant, muddy',
                'frequency_range': '800-2000 Hz presence'
            },
            7: {
                'name': 'Upper-Mid Formant',
                'tonal_quality': 'Nasal/Woody Quality',
                'description': 'Upper-mid resonance - nasal or woody character',
                'high_values': 'Nasal, woody, hollow (oboe, clarinet)',
                'low_values': 'Open, airy, spacious',
                'frequency_range': '2000-4000 Hz character'
            },
            8: {
                'name': 'High Formant',
                'tonal_quality': 'Brilliance/Sparkle',
                'description': 'High-frequency resonance - adds brilliance',
                'high_values': 'Brilliant, sparkling, crystalline sounds',
                'low_values': 'Dull, muffled, dark sounds',
                'frequency_range': '4000-8000 Hz brilliance'
            },
            9: {
                'name': 'Air/Breath Quality',
                'tonal_quality': 'Breathiness/Airiness',
                'description': 'High-frequency noise components - breath and air',
                'high_values': 'Breathy, airy (flute breath, vocal breath)',
                'low_values': 'Clean, pure tones without air noise',
                'frequency_range': '8000+ Hz air frequencies'
            },
            10: {
                'name': 'Sibilance/Texture',
                'tonal_quality': 'Roughness/Smoothness',
                'description': 'Very high-frequency texture information',
                'high_values': 'Rough, textured, sibilant, grainy',
                'low_values': 'Smooth, clean, polished',
                'frequency_range': 'Ultra-high frequency texture'
            },
            11: {
                'name': 'Fine Spectral Detail',
                'tonal_quality': 'Subtle Timbral Nuances',
                'description': 'Fine-grained spectral characteristics',
                'high_values': 'Complex, detailed, nuanced timbre',
                'low_values': 'Simple, basic, generic timbre',
                'frequency_range': 'Detailed spectral features'
            },
            12: {
                'name': 'Micro-Timbral Features',
                'tonal_quality': 'Instrument-Specific Character',
                'description': 'Subtle differences between similar instruments',
                'high_values': 'Distinctive, unique instrument character',
                'low_values': 'Generic, synthetic, processed sound',
                'frequency_range': 'Micro-spectral signatures'
            }
        }
        
        # Define typical instrument characteristics for reference
        self.instrument_profiles = {
            'violin': {'brightness': 'High', 'richness': 'High', 'attack': 'Medium', 'body': 'Medium'},
        }
        
    def load_audio(self, filepath):
        """Load audio file and return audio data and sample rate"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sr)
            return audio, sr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def extract_mfcc_features(self, audio, sr):
        """Extract comprehensive MFCC features from audio signal"""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Calculate comprehensive statistical measures
        mfcc_stats = {
            'mean': np.mean(mfccs, axis=1),
            'std': np.std(mfccs, axis=1),
            'min': np.min(mfccs, axis=1),
            'max': np.max(mfccs, axis=1),
            'median': np.median(mfccs, axis=1),
            'q25': np.percentile(mfccs, 25, axis=1),
            'q75': np.percentile(mfccs, 75, axis=1),
            'skewness': self._calculate_skewness(mfccs),
            'kurtosis': self._calculate_kurtosis(mfccs),
            'range': np.max(mfccs, axis=1) - np.min(mfccs, axis=1)
        }
        
        return mfccs, mfcc_stats
    
    def _calculate_skewness(self, data):
        """Calculate skewness for each MFCC coefficient"""
        return np.array([self._skew(row) for row in data])
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis for each MFCC coefficient"""
        return np.array([self._kurt(row) for row in data])
    
    def _skew(self, x):
        """Calculate skewness of a 1D array"""
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) / std) ** 3) if std != 0 else 0
    
    def _kurt(self, x):
        """Calculate kurtosis of a 1D array"""
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) / std) ** 4) - 3 if std != 0 else 0
    
    def analyze_tonal_qualities(self, mfcc_stats):
        """Analyze tonal qualities based on MFCC values with detailed interpretation"""
        tonal_analysis = {}
        
        for i in range(self.n_mfcc):
            if i in self.mfcc_tonal_qualities:
                mean_val = mfcc_stats['mean'][i]
                std_val = mfcc_stats['std'][i]
                
                # Improved intensity classification with coefficient-specific thresholds
                if i == 0:  # Energy coefficient - different scale
                    if mean_val > -5:
                        intensity = 'Very High'
                    elif mean_val > -15:
                        intensity = 'High'
                    elif mean_val > -25:
                        intensity = 'Medium'
                    elif mean_val > -35:
                        intensity = 'Low'
                    else:
                        intensity = 'Very Low'
                else:  # Other coefficients
                    if mean_val > 5:
                        intensity = 'Very High'
                    elif mean_val > 2:
                        intensity = 'High'
                    elif mean_val > -2:
                        intensity = 'Medium'
                    elif mean_val > -5:
                        intensity = 'Low'
                    else:
                        intensity = 'Very Low'
                
                # Stability analysis
                if std_val < 2:
                    stability = 'Very Stable'
                elif std_val < 5:
                    stability = 'Stable'
                elif std_val < 10:
                    stability = 'Variable'
                elif std_val < 20:
                    stability = 'Highly Variable'
                else:
                    stability = 'Extremely Variable'
                
                tonal_analysis[i] = {
                    'coefficient': f'MFCC {i}',
                    'name': self.mfcc_tonal_qualities[i]['name'],
                    'tonal_quality': self.mfcc_tonal_qualities[i]['tonal_quality'],
                    'mean_value': mean_val,
                    'std_value': std_val,
                    'intensity': intensity,
                    'stability': stability,
                    'description': self.mfcc_tonal_qualities[i]['description'],
                    'frequency_range': self.mfcc_tonal_qualities[i]['frequency_range']
                }
        
        return tonal_analysis
    
    def analyze_file(self, filepath):
        """Analyze a single audio file with comprehensive tonal analysis"""
        audio, sr = self.load_audio(filepath)
        if audio is None:
            return None
        
        # Extract MFCC features
        mfccs, mfcc_stats = self.extract_mfcc_features(audio, sr)
        
        # Analyze tonal qualities
        tonal_analysis = self.analyze_tonal_qualities(mfcc_stats)
        
        # Generate overall timbre profile
        timbre_profile = self.generate_timbre_profile(tonal_analysis)
        
        results = {
            'filename': Path(filepath).name,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'mfcc_coefficients': mfccs,
            'mfcc_statistics': mfcc_stats,
            'tonal_analysis': tonal_analysis,
            'timbre_profile': timbre_profile
        }
        
        return results
    
    def generate_timbre_profile(self, tonal_analysis):
        """Generate an overall timbre profile based on key MFCC coefficients"""
        profile = {}
        
        # Key timbral characteristics
        if 1 in tonal_analysis:  # Brightness
            profile['brightness'] = tonal_analysis[1]['intensity']
        if 3 in tonal_analysis:  # Harmonic richness
            profile['richness'] = tonal_analysis[3]['intensity']
        if 4 in tonal_analysis:  # Attack character
            profile['attack'] = tonal_analysis[4]['intensity']
        if 5 in tonal_analysis:  # Body/warmth
            profile['body'] = tonal_analysis[5]['intensity']
        if 6 in tonal_analysis:  # Presence
            profile['presence'] = tonal_analysis[6]['intensity']
        if 8 in tonal_analysis:  # Brilliance
            profile['brilliance'] = tonal_analysis[8]['intensity']
        if 9 in tonal_analysis:  # Breathiness
            profile['breathiness'] = tonal_analysis[9]['intensity']
        if 10 in tonal_analysis:  # Roughness
            profile['roughness'] = tonal_analysis[10]['intensity']
        
        return profile
    
    def print_detailed_analysis(self, result):
        """Print comprehensive tonal analysis for a single file"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TONAL ANALYSIS: {result['filename']}")
        print(f"{'='*80}")
        print(f"Duration: {result['duration']:.2f} seconds | Sample Rate: {result['sample_rate']} Hz")
        print(f"\n{'DETAILED MFCC COEFFICIENT ANALYSIS'}")
        print(f"{'-'*80}")
        
        for i in range(self.n_mfcc):
            if i in result['tonal_analysis']:
                analysis = result['tonal_analysis'][i]
                print(f"\n{analysis['coefficient']}: {analysis['name']}")
                print(f"  Tonal Quality: {analysis['tonal_quality']}")
                print(f"  Frequency Range: {analysis['frequency_range']}")
                print(f"  Description: {analysis['description']}")
                print(f"  Mean Value: {analysis['mean_value']:.3f} ({analysis['intensity']})")
                print(f"  Variability: {analysis['std_value']:.3f} ({analysis['stability']})")
                
                # Add interpretation based on intensity
                if analysis['intensity'] in ['High', 'Very High']:
                    interp = self.mfcc_tonal_qualities[i]['high_values']
                else:
                    interp = self.mfcc_tonal_qualities[i]['low_values']
                print(f"  Interpretation: {interp}")
        
        # Overall timbre profile
        print(f"\n{'='*80}")
        print("OVERALL TIMBRE PROFILE:")
        print(f"{'='*80}")
        
        profile = result['timbre_profile']
        print(f"🔆 Brightness: {profile.get('brightness', 'N/A')}")
        print(f"🎵 Harmonic Richness: {profile.get('richness', 'N/A')}")
        print(f"⚡ Attack Character: {profile.get('attack', 'N/A')}")
        print(f"💪 Body/Warmth: {profile.get('body', 'N/A')}")
        print(f"📢 Presence/Clarity: {profile.get('presence', 'N/A')}")
        print(f"✨ Brilliance: {profile.get('brilliance', 'N/A')}")
        print(f"💨 Breathiness: {profile.get('breathiness', 'N/A')}")
        print(f"🏔️ Roughness: {profile.get('roughness', 'N/A')}")
        
        print(f"\n{'='*80}")
    
    def analyze_directory(self, directory_path):
        """Analyze all .wav files in a directory"""
        directory = Path(directory_path)
        wav_files = list(directory.glob("*.wav"))
        
        if not wav_files:
            print(f"No .wav files found in {directory_path}")
            return []
        
        results = []
        for wav_file in wav_files:
            print(f"Analyzing: {wav_file.name}")
            result = self.analyze_file(wav_file)
            if result:
                results.append(result)
        
        return results
    
    def compare_tonal_qualities(self, results):
        """Compare tonal qualities between multiple files"""
        if len(results) < 2:
            print("Need at least 2 files for comparison")
            return
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE TONAL QUALITY COMPARISON")
        print(f"{'='*100}")
        
        # Create detailed comparison table
        qualities = ['Brightness', 'Richness', 'Attack', 'Body', 'Presence', 'Brilliance', 'Breathiness', 'Roughness']
        
        print(f"{'File':<25}", end='')
        for quality in qualities:
            print(f"{quality:<12}", end='')
        print()
        print('-' * (25 + 12 * len(qualities)))
        
        for result in results:
            filename = result['filename'][:23]
            print(f"{filename:<25}", end='')
            
            profile = result['timbre_profile']
            for quality in qualities:
                value = profile.get(quality.lower(), 'N/A')
                print(f"{value:<12}", end='')
            print()
        
        print(f"\n{'='*100}")
    
    def create_comprehensive_dataframe(self, results):
        """Create a comprehensive DataFrame with all MFCC features and tonal qualities"""
        if not results:
            return pd.DataFrame()
        
        data = []
        for result in results:
            row = {
                'filename': result['filename'],
                'duration': result['duration'],
                'sample_rate': result['sample_rate']
            }
            
            # Add all MFCC statistics
            for stat_name, stat_values in result['mfcc_statistics'].items():
                for i, value in enumerate(stat_values):
                    row[f'mfcc_{i}_{stat_name}'] = value
            
            # Add tonal quality interpretations
            for i in range(self.n_mfcc):
                if i in result['tonal_analysis']:
                    analysis = result['tonal_analysis'][i]
                    row[f'mfcc_{i}_name'] = analysis['name']
                    row[f'mfcc_{i}_quality'] = analysis['tonal_quality']
                    row[f'mfcc_{i}_intensity'] = analysis['intensity']
                    row[f'mfcc_{i}_stability'] = analysis['stability']
            
            # Add timbre profile
            for key, value in result['timbre_profile'].items():
                row[f'timbre_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_comprehensive_analysis(self, results, max_files=4):
        """Create comprehensive visualization of all 13 MFCC coefficients and tonal qualities"""
        if not results:
            print("No results to plot")
            return
        
        plot_results = results[:max_files]
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Comprehensive MFCC Tonal Analysis Dashboard', fontsize=24, fontweight='bold')
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_results)))
        
        # Plot 1: All 13 MFCC coefficients comparison
        ax1 = fig.add_subplot(gs[0, :2])
        x_pos = np.arange(self.n_mfcc)
        width = 0.8 / len(plot_results)
        
        for i, (result, color) in enumerate(zip(plot_results, colors)):
            means = result['mfcc_statistics']['mean']
            offset = (i - len(plot_results)/2) * width
            bars = ax1.bar(x_pos + offset, means, width, label=result['filename'][:15], 
                          color=color, alpha=0.8)
        
        ax1.set_title('All 13 MFCC Coefficients - Mean Values', fontsize=16, fontweight='bold')
        ax1.set_xlabel('MFCC Coefficient')
        ax1.set_ylabel('Mean Value')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tonal Quality Radar Chart
        ax2 = fig.add_subplot(gs[0, 2:], projection='polar')
        if plot_results:
            for result, color in zip(plot_results[:2], colors[:2]):  # Max 2 for clarity
                angles = np.linspace(0, 2*np.pi, self.n_mfcc, endpoint=False).tolist()
                angles += angles[:1]
                
                means = result['mfcc_statistics']['mean']
                # Normalize for radar chart
                normalized = [(val + 50) / 100 for val in means]
                normalized += normalized[:1]
                
                ax2.plot(angles, normalized, 'o-', linewidth=2, color=color, 
                        label=result['filename'][:15])
                ax2.fill(angles, normalized, alpha=0.2, color=color)
            
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
            ax2.set_title('MFCC Tonal Profile Comparison', fontsize=16, fontweight='bold')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax2.grid(True)
        
        # Plot 3: Key Tonal Qualities
        ax3 = fig.add_subplot(gs[1, :2])
        key_qualities = ['Brightness', 'Richness', 'Attack', 'Body', 'Presence', 'Brilliance']
        key_mfccs = [1, 3, 4, 5, 6, 8]
        
        x_pos = np.arange(len(key_qualities))
        width = 0.8 / len(plot_results)
        
        for i, (result, color) in enumerate(zip(plot_results, colors)):
            values = [result['mfcc_statistics']['mean'][mfcc_idx] for mfcc_idx in key_mfccs]
            offset = (i - len(plot_results)/2) * width
            ax3.bar(x_pos + offset, values, width, label=result['filename'][:15], 
                   color=color, alpha=0.8)
        
        ax3.set_title('Key Tonal Qualities Comparison', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Tonal Quality')
        ax3.set_ylabel('MFCC Value')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(key_qualities, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: MFCC Stability (Standard Deviations)
        ax4 = fig.add_subplot(gs[1, 2:])
        for result, color in zip(plot_results, colors):
            stds = result['mfcc_statistics']['std']
            ax4.plot(range(self.n_mfcc), stds, 'o-', color=color, linewidth=2,
                    markersize=6, label=result['filename'][:15])
        
        ax4.set_title('MFCC Stability Analysis', fontsize=16, fontweight='bold')
        ax4.set_xlabel('MFCC Coefficient')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_xticks(range(self.n_mfcc))
        ax4.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Detailed MFCC Heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        if plot_results:
            mfccs = plot_results[0]['mfcc_coefficients']
            im = ax5.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis')
            ax5.set_title(f'MFCC Time Evolution: {plot_results[0]["filename"][:20]}', 
                         fontsize=16, fontweight='bold')
            ax5.set_xlabel('Time Frame')
            ax5.set_ylabel('MFCC Coefficient')
            ax5.set_yticks(range(self.n_mfcc))
            ax5.set_yticklabels([f'C{i}' for i in range(self.n_mfcc)])
            plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # Plot 6: Timbre Profile Comparison
        ax6 = fig.add_subplot(gs[2, 2:])
        if len(plot_results) >= 2:
            intensity_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
            
            for result, color in zip(plot_results, colors):
                intensities = []
                for mfcc_idx in range(self.n_mfcc):
                    if mfcc_idx in result['tonal_analysis']:
                        intensity = result['tonal_analysis'][mfcc_idx]['intensity']
                        intensities.append(intensity_map.get(intensity, 0))
                    else:
                        intensities.append(0)
                
                ax6.plot(range(self.n_mfcc), intensities, 'o-', linewidth=2, 
                        markersize=8, color=color, label=result['filename'][:15])
            
            ax6.set_title('Tonal Intensity Profile', fontsize=16, fontweight='bold')
            ax6.set_xlabel('MFCC Coefficient')
            ax6.set_ylabel('Intensity Level')
            ax6.set_xticks(range(self.n_mfcc))
            ax6.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
            ax6.set_yticks([1, 2, 3, 4, 5])
            ax6.set_yticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Statistical Distribution
        ax7 = fig.add_subplot(gs[3, :2])
        if plot_results:
            result = plot_results[0]
            stats = result['mfcc_statistics']
            
            # Box plot style data
            positions = range(self.n_mfcc)
            means = stats['mean']
            stds = stats['std']
            
            ax7.errorbar(positions, means, yerr=stds, fmt='o', capsize=5, capthick=2)
            ax7.set_title(f'MFCC Statistical Distribution: {result["filename"][:20]}', 
                         fontsize=16, fontweight='bold')
            ax7.set_xlabel('MFCC Coefficient')
            ax7.set_ylabel('Value')
            ax7.set_xticks(positions)
            ax7.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Frequency Range Visualization
        ax8 = fig.add_subplot(gs[3, 2:])
        freq_ranges = ['Full', 'Weighted', 'High', 'Harmonic', 'Temporal', 
                      '200-800Hz', '800-2kHz', '2-4kHz', '4-8kHz', '8kHz+', 
                      'Ultra-high', 'Detail', 'Micro']
        
        if plot_results:
            result = plot_results[0]
            values = [result['mfcc_statistics']['mean'][i] for i in range(self.n_mfcc)]
            
            bars = ax8.barh(range(self.n_mfcc), values, color=colors[0], alpha=0.7)
            ax8.set_title('MFCC Frequency Range Analysis', fontsize=16, fontweight='bold')
            ax8.set_xlabel('MFCC Mean Value')
            ax8.set_ylabel('MFCC Coefficient')
            ax8.set_yticks(range(self.n_mfcc))
            ax8.set_yticklabels([f'C{i}: {freq_ranges[i]}' for i in range(self.n_mfcc)])
            ax8.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax8.text(width + 0.1 if width >= 0 else width - 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)

        # Plot 9: MFCC Dynamic Range Comparison
        ax9 = fig.add_subplot(gs[4, :2])
        if plot_results:
            x_pos = np.arange(self.n_mfcc)
            width = 0.8 / len(plot_results)
            
            for i, (result, color) in enumerate(zip(plot_results, colors)):
                ranges = result['mfcc_statistics']['range']
                offset = (i - len(plot_results)/2) * width
                bars = ax9.bar(x_pos + offset, ranges, width, label=result['filename'][:15], 
                              color=color, alpha=0.8)
            
            ax9.set_title('MFCC Dynamic Range Comparison', fontsize=16, fontweight='bold')
            ax9.set_xlabel('MFCC Coefficient')
            ax9.set_ylabel('Range (Max - Min)')
            ax9.set_xticks(x_pos)
            ax9.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            # Add coefficient names as secondary labels
            ax9_twin = ax9.twiny()
            ax9_twin.set_xlim(ax9.get_xlim())
            ax9_twin.set_xticks(x_pos)
            coeff_names = [self.mfcc_tonal_qualities[i]['name'][:8] + '...' if len(self.mfcc_tonal_qualities[i]['name']) > 8 
                          else self.mfcc_tonal_qualities[i]['name'] for i in range(self.n_mfcc)]
            ax9_twin.set_xticklabels(coeff_names, rotation=45, ha='left', fontsize=8)

        # Plot 10: Statistical Shape Analysis (Skewness & Kurtosis)
        ax10 = fig.add_subplot(gs[4, 2:])
        if plot_results:
            # Use first result for shape analysis, or compare multiple if available
            if len(plot_results) == 1:
                # Single file: show both skewness and kurtosis
                result = plot_results[0]
                skewness = result['mfcc_statistics']['skewness']
                kurtosis = result['mfcc_statistics']['kurtosis']
                
                x = np.arange(self.n_mfcc)
                width = 0.35
                
                bars1 = ax10.bar(x - width/2, skewness, width, label='Skewness', 
                               alpha=0.7, color='skyblue')
                bars2 = ax10.bar(x + width/2, kurtosis, width, label='Kurtosis', 
                               alpha=0.7, color='lightcoral')
                
                ax10.set_title(f'Statistical Shape Analysis: {result["filename"][:20]}', 
                             fontsize=16, fontweight='bold')
                ax10.set_xlabel('MFCC Coefficient')
                ax10.set_ylabel('Statistical Value')
                ax10.set_xticks(x)
                ax10.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
                ax10.legend()
                ax10.grid(True, alpha=0.3)
                ax10.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
                
            else:
                # Multiple files: compare skewness across files
                x_pos = np.arange(self.n_mfcc)
                width = 0.8 / len(plot_results)
                
                for i, (result, color) in enumerate(zip(plot_results, colors)):
                    skewness = result['mfcc_statistics']['skewness']
                    offset = (i - len(plot_results)/2) * width
                    ax10.bar(x_pos + offset, skewness, width, label=result['filename'][:15], 
                           color=color, alpha=0.8)
                
                ax10.set_title('Skewness Comparison Across Files', fontsize=16, fontweight='bold')
                ax10.set_xlabel('MFCC Coefficient')
                ax10.set_ylabel('Skewness')
                ax10.set_xticks(x_pos)
                ax10.set_xticklabels([f'C{i}' for i in range(self.n_mfcc)])
                ax10.legend()
                ax10.grid(True, alpha=0.3)
                ax10.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

        plt.tight_layout()
        plt.show()

    