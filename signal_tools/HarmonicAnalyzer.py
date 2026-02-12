"""
Author: Joseph Moreno
Description: Python module that assists with harmonic analysis of sampled signals. One function specifically computes the strength of even harmonic peaks (dB),
while another function calculates harmonic frequencies and their corresponding Nyquist zones.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_even_harmonics(filename):
    with np.load(filename) as data_file:
        signal = data_file['data'].flatten().astype(float)
        fs = data_file['sample_rate']
        
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs_full = np.fft.fftfreq(n, 1/fs)
    
    # Find fundamental frequency f0
    pos_fft = np.abs(fft_signal[:n//2])
    pos_freqs = freqs_full[:n//2]
    fundamental_idx = np.argmax(pos_fft[1:]) + 1
    f0 = pos_freqs[fundamental_idx]
    
    # Calculate half-period phase shift
    shift_samples = fs / (2 * f0)
    norm_freqs = np.fft.fftfreq(n)
    phase_shift = np.exp(-2j * np.pi * norm_freqs * shift_samples)
    
    # Summed signal
    summed_fft = fft_signal + (fft_signal * phase_shift)
    
    # Spectral magnitudes
    freqs_r = np.fft.rfftfreq(n, 1/fs)
    mag_sum_db = 20 * np.log10(np.abs(summed_fft[:n//2+1]) + 1e-9)
    mag_orig_db = 20 * np.log10(np.abs(fft_signal[:n//2+1]) + 1e-9)
    
    # Identify even harmonic frequencies
    even_mults = [2, 4, 6, 8, 10, 12]
    even_freqs = [m * f0 for m in even_mults if m * f0 < fs/2]
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_r / 1e6, mag_sum_db, color='red', label='Summed Signal (Red Line)')
    
    # Highlight even harmonic peaks
    print("Even Harmonics in Summed Signal:")
    for ef in even_freqs:
        idx = np.argmin(np.abs(freqs_r - ef))
        val = mag_sum_db[idx]
        orig_val = mag_orig_db[idx]
        print(f"  {ef/1e6:.1f} MHz: {val:.2f} dB (Original: {orig_val:.2f} dB)")
        
        plt.plot(ef/1e6, val, 'ko') # Black dot
        plt.annotate(f'{ef/1e6:.1f} MHz\n({val:.1f} dB)', 
                     xy=(ef/1e6, val), 
                     xytext=(ef/1e6, val+15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                     ha='center')

    plt.title("Summed Spectrum: Locating the Hidden Even Harmonics")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig('even_harmonics_labeled.png')
    return f0

def calculate_harmonic_aliases(f_sig, fs, num_harmonics=12):
    fn = fs / 2
    print(f"{'Harmonic':<10} | {'Real Freq (MHz)':<16} | {'Zone':<6} | {'Alias (MHz)':<12} | {'Type'}")
    print("-" * 65)
    
    for k in range(1, num_harmonics + 1):
        f_k = k * f_sig
        
        # Calculate Nyquist Zone: Zone n contains ((n-1)*fn, n*fn]
        zone = int(np.ceil(f_k / fn))
        
        # Calculate Alias Frequency in the first Nyquist Zone (0 to fs/2)
        f_rem = f_k % fs
        if f_rem > fn:
            f_alias = fs - f_rem
            spectral_type = "Inverted"
        else:
            f_alias = f_rem
            spectral_type = "Normal"
            
        print(f"{k:<10} | {f_k:<16.2f} | {zone:<6} | {f_alias:<12.3f} | {spectral_type}")
