"""
Author: Joseph Moreno
Description: Takes two file paths for sampled sine wave data as inputs, one with anti-aliasing filters and the other without. The functions below plot the two data sets on top of one another to compare time domain and frequency domain (voltage, power spectra) graphs.
"""
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows

class SignalAnalyzer:
    """
    Creates an analyzer object for a specific filtered/bypass signal measurement
    """
    def __init__(self, bypassed_path, filtered_path):
        self.b_data = np.load(bypassed_path)
        self.f_data = np.load(filtered_path)
        self.fs = float(self.b_data['sample_rate'])  
        self.n = int(self.b_data['nsamples'])
        self.target_f = float(self.b_data['signal_freq'])
        
        # Load sampled data from loaded files for processing
        self.y_b = self.b_data['data'][0].astype(float)
        self.y_f = self.f_data['data'][0].astype(float)
        self.t = np.arange(self.n) / self.fs

    def plot_voltage_spectrum(self, save_path='voltage_spectrum.png'):
        """
        Produces a linearly scaled voltage spectrum using (Hann) windowed data
        """
        xf = fftfreq(self.n, 1/self.fs)[:self.n//2]
        win = windows.hann(self.n)
        norm = 2.0 / np.sum(win) # Normalization for Hann single-sided FFT
        
        # Applying Hann window to signal and removing dc offset by subtracting the mean
        v_b = np.abs(fft((self.y_b - np.mean(self.y_b)) * win)[:self.n//2]) * norm
        v_f = np.abs(fft((self.y_f - np.mean(self.y_f)) * win)[:self.n//2]) * norm
        
        plt.figure(figsize=(10, 5))
        plt.plot(xf / 1e6, v_b, label='Bypassed (Noisy)', color='red', alpha=0.7)
        plt.plot(xf / 1e6, v_f, label='Filtered (Clean)', color='tab:blue', linewidth=1.5)
        
        plt.title(f'Linear Voltage Spectrum @ {self.target_f/1e6} MHz', fontsize=12)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (Linear ADC Units)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        return save_path
        
    def plot_comparison(self, rf=1, save_path='comparison_plot.png'):
        """
        Overlays time domain graphs for a filtered and bypassed signal of a certain radio frequency (same sampling rate) 
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 1. time series graph
        zoom_idx = int(self.fs / 1e6 * 60) 
        ax1.plot(self.t[:zoom_idx] * 1e6, self.y_b[:zoom_idx], 
                 label='Bypassed (Unfiltered)', color='tab:red', alpha=0.7)
        ax1.plot(self.t[:zoom_idx] * 1e6, self.y_f[:zoom_idx], 
                 label='Filtered (Anti-Aliased)', color='tab:blue', linewidth=2)
        ax1.set_title(f'Time Domain: {rf} MHz Sine Wave Comparison', fontsize=14)
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.set_ylabel('ADC Counts (int8)')
        ax1.legend()

        # frequency domain plot obtained via Fast Fourier Transform
        xf = fftfreq(self.n, 1/self.fs)[:self.n//2]
        psd_b = 20 * np.log10(np.abs(fft(self.y_b)[:self.n//2]) + 1e-6)
        psd_f = 20 * np.log10(np.abs(fft(self.y_f)[:self.n//2]) + 1e-6)

        ax2.plot(xf / 1e6, psd_b, label='Bypassed Spectrum', color='tab:red', alpha=0.5)
        ax2.plot(xf / 1e6, psd_f, label='Filtered Spectrum', color='tab:blue')
        ax2.axvline(x=self.fs/2e6, color='black', linestyle='--', label='Nyquist Limit')
        ax2.axvline(x=abs(self.fs - self.target_f)/1e6, color='green', linestyle='--', label='Theoretical Alias Peak')
        
        ax2.set_title('Frequency Domain: Power Spectral Density', fontsize=14)
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_ylim(bottom=0)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    def print_stats(self):
        """
        Prints basic signal statistics
        """
        print(f"{'Metric':<20} | {'Bypassed':<10} | {'Filtered':<10}")
        print("-" * 45)
        print(f"{'Mean (Offset)':<20} | {np.mean(self.y_b):>10.2f} | {np.mean(self.y_f):>10.2f}")
        print(f"{'Std Dev (RMS)':<20} | {np.std(self.y_b):>10.2f} | {np.std(self.y_f):>10.2f}")
        print(f"{'Peak-to-Peak':<20} | {np.ptp(self.y_b):>10.2f} | {np.ptp(self.y_f):>10.2f}")
