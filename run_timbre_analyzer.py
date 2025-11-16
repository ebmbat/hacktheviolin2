#!/usr/bin/env python3

from comprehensive_timbre_analyzer import ComprehensiveTimbreAnalyzer
import sys
from pathlib import Path

def main():
    # Initialize analyzer with custom parameters if needed
    analyzer = ComprehensiveTimbreAnalyzer(
        n_mfcc=13,      # Number of MFCC coefficients
        sr=22050,       # Sample rate
        hop_length=512, # Frame hop length
        n_fft=2048     # FFT window size
    )
    
    # Check if directory argument provided
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    else:
        audio_dir = "audio_samples"  # Default directory
    
    # Verify directory exists
    if not Path(audio_dir).exists():
        print(f"Error: Directory '{audio_dir}' not found!")
        print("Please create the directory and add .wav files")
        return
    
    print(f"Analyzing audio files in: {audio_dir}")
    print("=" * 60)
    
    # Analyze all files in directory
    results = analyzer.analyze_directory(audio_dir)
    
    if not results:
        print("No .wav files found or analysis failed!")
        return
    
    print(f"\nSuccessfully analyzed {len(results)} files:")
    for result in results:
        print(f"  - {result['filename']} ({result['duration']:.2f}s)")
    
    # Generate comprehensive analysis
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # 1. Individual detailed analyses
    for result in results:
        analyzer.print_detailed_analysis(result)
    
    # 2. Comparative analysis
    if len(results) > 1:
        analyzer.compare_tonal_qualities(results)
    
    # 3. Create DataFrame and save to CSV
    df = analyzer.create_comprehensive_dataframe(results)
    csv_filename = "timbre_analysis_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # 4. Generate comprehensive visualization dashboard
    print("\nGenerating visualization dashboard...")
    analyzer.plot_comprehensive_analysis(results)
    print("Dashboard complete! Close the plot window to continue.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()