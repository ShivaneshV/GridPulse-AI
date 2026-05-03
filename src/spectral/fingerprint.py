"""
Spectral Fingerprinting Engine - GridPulse AI
==============================================
Uses Fast Fourier Transforms (FFT) to detect illegal power taps by identifying
specific harmonic distortion signatures in the grid frequency spectrum.

The core insight: Illegal bypass taps create specific Harmonic Distortion and
Reactive Power signatures that can be mathematically isolated from normal 50Hz
grid load using spectral analysis.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from typing import Tuple, Dict, List, Optional
import warnings


class SpectralFingerprinter:
    """
    Core spectral fingerprinting engine for detecting power theft through
    harmonic analysis of electrical signals.
    """
    
    def __init__(self, 
                 sampling_rate: float = 1000.0,  # Hz
                 fundamental_freq: float = 50.0,  # Hz (India grid frequency)
                 fft_size: int = 4096,
                 detection_threshold: float = 3.0):  # Standard deviations
        """
        Initialize the Spectral Fingerprinter.
        
        Args:
            sampling_rate: Sampling rate in Hz (default 1000Hz for 1kHz)
            fundamental_freq: Grid fundamental frequency (50Hz for India)
            fft_size: Size of FFT window
            detection_threshold: Number of standard deviations for anomaly detection
        """
        self.sampling_rate = sampling_rate
        self.fundamental_freq = fundamental_freq
        self.fft_size = fft_size
        self.detection_threshold = detection_threshold
        
        # Pre-compute frequency bins for harmonic analysis
        self.freq_resolution = sampling_rate / fft_size
        self.harmonic_bins = self._compute_harmonic_bins()
        
        # Baseline spectral signature (to be calibrated)
        self.baseline_signature = None
        self.baseline_std = None
        
    def _compute_harmonic_bins(self) -> Dict[int, Tuple[int, int]]:
        """Compute frequency bin ranges for harmonic analysis."""
        harmonic_bins = {}
        for n in range(1, 21):  # Analyze up to 20th harmonic
            center_freq = n * self.fundamental_freq
            bin_center = int(center_freq / self.freq_resolution)
            # ±2 bins around harmonic center
            start_bin = max(0, bin_center - 2)
            end_bin = min(self.fft_size // 2, bin_center + 2)
            harmonic_bins[n] = (start_bin, end_bin)
        return harmonic_bins
    
    def compute_spectrum(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency spectrum of the input signal.
        
        Args:
            signal_data: Time-domain electrical signal
            
        Returns:
            frequencies: Array of frequency bins
            magnitudes: Magnitude spectrum
        """
        # Apply Hanning window to reduce spectral leakage
        windowed = signal_data * np.hanning(len(signal_data))
        
        # Zero-pad to FFT size if needed
        if len(windowed) < self.fft_size:
            windowed = np.pad(windowed, (0, self.fft_size - len(windowed)), 'constant')
        
        # Compute FFT
        fft_values = fft(windowed[:self.fft_size])
        frequencies = fftfreq(self.fft_size, 1/self.sampling_rate)[:self.fft_size//2]
        magnitudes = 2.0 / self.fft_size * np.abs(fft_values[:self.fft_size//2])
        
        return frequencies, magnitudes
    
    def extract_spectral_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive spectral features for theft detection.
        
        Returns:
            Dictionary of spectral features including:
            - Total Harmonic Distortion (THD)
            - Individual harmonic magnitudes
            - Inter-harmonic content
            - Spectral flatness
            - Peak-to-average ratio
        """
        frequencies, magnitudes = self.compute_spectrum(signal_data)
        
        features = {}
        
        # 1. Fundamental frequency magnitude
        fundamental_bin = self.harmonic_bins[1]
        fundamental_mag = np.mean(magnitudes[fundamental_bin[0]:fundamental_bin[1]])
        features['fundamental_magnitude'] = fundamental_mag
        
        # 2. Total Harmonic Distortion (THD)
        harmonic_mags = []
        for n in range(2, 11):  # 2nd to 10th harmonic
            bin_range = self.harmonic_bins[n]
            harmonic_mag = np.mean(magnitudes[bin_range[0]:bin_range[1]])
            harmonic_mags.append(harmonic_mag)
            features[f'harmonic_{n}'] = harmonic_mag
        
        # THD calculation
        thd = np.sqrt(sum(h**2 for h in harmonic_mags[1:])) / harmonic_mags[0] * 100
        features['total_harmonic_distortion'] = thd
        
        # 3. Inter-harmonic content (between harmonics)
        inter_harmonic_power = 0
        for i in range(len(self.harmonic_bins) - 1):
            n1 = i + 1
            n2 = i + 2
            if n1 in self.harmonic_bins and n2 in self.harmonic_bins:
                start = self.harmonic_bins[n1][1]
                end = self.harmonic_bins[n2][0]
                if start < end:
                    inter_harmonic_power += np.sum(magnitudes[start:end]**2)
        features['inter_harmonic_power'] = inter_harmonic_power
        
        # 4. Spectral flatness (measure of tonality vs noise)
        geometric_mean = np.exp(np.mean(np.log(magnitudes[magnitudes > 0])))
        arithmetic_mean = np.mean(magnitudes)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # 5. Peak-to-average ratio
        features['peak_to_average'] = np.max(magnitudes) / np.mean(magnitudes)
        
        # 6. Reactive power signature (phase analysis)
        features['reactive_power_signature'] = self._compute_reactive_signature(signal_data)
        
        # 7. High-frequency noise floor (indicates arcing/illegal connections)
        high_freq_start = int(500 / self.freq_resolution)  # Above 500Hz
        features['high_freq_noise_floor'] = np.mean(magnitudes[high_freq_start:])
        
        return features
    
    def _compute_reactive_signature(self, signal_data: np.ndarray) -> float:
        """
        Compute reactive power signature using Hilbert transform.
        Illegal taps often show distinctive reactive power patterns.
        """
        # Analytic signal via Hilbert transform
        analytic_signal = signal.hilbert(signal_data)
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Reactive power is related to phase variations
        phase_derivative = np.diff(instantaneous_phase)
        reactive_signature = np.std(phase_derivative)
        
        return reactive_signature
    
    def detect_theft_signature(self, 
                              feeder_signal: np.ndarray, 
                              meter_signals: List[np.ndarray],
                              confidence_threshold: float = 0.85) -> Dict:
        """
        Detect theft by comparing feeder-level spectrum with sum of meter spectra.
        
        The core logic: If the feeder shows signatures (like resistive load harmonics)
        that aren't reflected in the downstream meter data, it indicates bypass theft.
        
        Args:
            feeder_signal: Signal from feeder-level monitoring
            meter_signals: List of signals from individual meters
            confidence_threshold: Confidence threshold for theft detection
            
        Returns:
            Dictionary with detection results and confidence scores
        """
        # Extract features from feeder
        feeder_features = self.extract_spectral_features(feeder_signal)
        
        # Extract features from each meter and aggregate
        meter_features_list = [self.extract_spectral_features(meter) for meter in meter_signals]
        
        # Aggregate meter features (sum for power-related, mean for ratios)
        aggregated_meter_features = {}
        for key in feeder_features.keys():
            if 'power' in key.lower() or 'magnitude' in key.lower():
                # Sum for power-related features
                aggregated_meter_features[key] = sum(feat[key] for feat in meter_features_list)
            else:
                # Weighted average for ratio features
                aggregated_meter_features[key] = np.mean([feat[key] for feat in meter_features_list])
        
        # Compute discrepancy scores
        discrepancy_scores = self._compute_discrepancy(feeder_features, aggregated_meter_features)
        
        # Detect specific theft signatures
        theft_signatures = self._identify_theft_signatures(feeder_features, aggregated_meter_features)
        
        # Calculate overall confidence
        confidence = self._calculate_theft_confidence(discrepancy_scores, theft_signatures)
        
        return {
            'theft_detected': confidence >= confidence_threshold,
            'confidence': confidence,
            'discrepancy_scores': discrepancy_scores,
            'theft_signatures': theft_signatures,
            'feeder_features': feeder_features,
            'meter_features': aggregated_meter_features
        }
    
    def _compute_discrepancy(self, feeder: Dict, meters: Dict) -> Dict[str, float]:
        """Compute discrepancy between feeder and aggregated meter readings."""
        discrepancies = {}
        
        # Energy balance discrepancy
        if 'fundamental_magnitude' in feeder and 'fundamental_magnitude' in meters:
            energy_discrepancy = abs(feeder['fundamental_magnitude'] - meters['fundamental_magnitude'])
            discrepancies['energy_balance'] = energy_discrepancy / max(feeder['fundamental_magnitude'], 1e-10)
        
        # Harmonic discrepancy (theft often adds specific harmonics)
        for n in range(2, 11):
            key = f'harmonic_{n}'
            if key in feeder and key in meters:
                harmonic_disc = abs(feeder[key] - meters[key])
                discrepancies[f'harmonic_{n}_discrepancy'] = harmonic_disc / max(feeder[key], 1e-10)
        
        # THD discrepancy
        if 'total_harmonic_distortion' in feeder and 'total_harmonic_distortion' in meters:
            thd_disc = abs(feeder['total_harmonic_distortion'] - meters['total_harmonic_distortion'])
            discrepancies['thd_discrepancy'] = thd_disc
        
        # High-frequency noise discrepancy (indicates arcing/illegal connections)
        if 'high_freq_noise_floor' in feeder and 'high_freq_noise_floor' in meters:
            noise_disc = abs(feeder['high_freq_noise_floor'] - meters['high_freq_noise_floor'])
            discrepancies['noise_floor_discrepancy'] = noise_disc / max(feeder['high_freq_noise_floor'], 1e-10)
        
        return discrepancies
    
    def _identify_theft_signatures(self, feeder: Dict, meters: Dict) -> Dict[str, bool]:
        """Identify specific theft signatures based on domain knowledge."""
        signatures = {}
        
        # Signature 1: Resistive load hooking (high 3rd harmonic)
        if 'harmonic_3_discrepancy' in self._compute_discrepancy(feeder, meters):
            h3_disc = self._compute_discrepancy(feeder, meters)['harmonic_3_discrepancy']
            signatures['resistive_hooking'] = h3_disc > 0.3  # 30% discrepancy threshold
        
        # Signature 2: Arcing signature (high-frequency noise)
        if 'noise_floor_discrepancy' in self._compute_discrepancy(feeder, meters):
            noise_disc = self._compute_discrepancy(feeder, meters)['noise_floor_discrepancy']
            signatures['arcing_signature'] = noise_disc > 0.5  # 50% discrepancy threshold
        
        # Signature 3: Phase imbalance (from reactive signature)
        if 'reactive_power_signature' in feeder and 'reactive_power_signature' in meters:
            reactive_disc = abs(feeder['reactive_power_signature'] - meters['reactive_power_signature'])
            signatures['phase_imbalance'] = reactive_disc > 0.2
        
        # Signature 4: Spectral flatness change (indicates non-linear loads)
        if 'spectral_flatness' in feeder and 'spectral_flatness' in meters:
            flatness_disc = abs(feeder['spectral_flatness'] - meters['spectral_flatness'])
            signatures['non_linear_load'] = flatness_disc > 0.15
        
        return signatures
    
    def _calculate_theft_confidence(self, 
                                   discrepancies: Dict[str, float], 
                                   signatures: Dict[str, bool]) -> float:
        """
        Calculate overall theft confidence using weighted combination of
        discrepancies and detected signatures.
        """
        confidence = 0.0
        
        # Weight discrepancies (40% of total confidence)
        if discrepancies:
            disc_weights = {
                'energy_balance': 0.3,
                'thd_discrepancy': 0.25,
                'noise_floor_discrepancy': 0.25,
                'harmonic_3_discrepancy': 0.2
            }
            
            weighted_disc = 0.0
            total_weight = 0.0
            for key, weight in disc_weights.items():
                if key in discrepancies:
                    # Normalize discrepancy to 0-1 range
                    normalized_disc = min(discrepancies[key] / 0.5, 1.0)
                    weighted_disc += normalized_disc * weight
                    total_weight += weight
            
            if total_weight > 0:
                confidence += 0.4 * (weighted_disc / total_weight)
        
        # Weight signatures (60% of total confidence)
        if signatures:
            sig_weights = {
                'resistive_hooking': 0.3,
                'arcing_signature': 0.3,
                'phase_imbalance': 0.2,
                'non_linear_load': 0.2
            }
            
            weighted_sig = 0.0
            total_weight = 0.0
            for key, weight in sig_weights.items():
                if key in signatures:
                    weighted_sig += (1.0 if signatures[key] else 0.0) * weight
                    total_weight += weight
            
            if total_weight > 0:
                confidence += 0.6 * (weighted_sig / total_weight)
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def calibrate_baseline(self, normal_signals: List[np.ndarray]):
        """
        Calibrate baseline spectral signature using normal (non-theft) signals.
        
        Args:
            normal_signals: List of signals known to be from normal operation
        """
        features_list = [self.extract_spectral_features(sig) for sig in normal_signals]
        
        # Compute mean and std for each feature
        feature_arrays = {key: [] for key in features_list[0].keys()}
        for features in features_list:
            for key, value in features.items():
                feature_arrays[key].append(value)
        
        self.baseline_signature = {key: np.mean(values) for key, values in feature_arrays.items()}
        self.baseline_std = {key: np.std(values) for key, values in feature_arrays.items()}
    
    def is_anomalous(self, signal_data: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Check if a signal is anomalous compared to calibrated baseline.
        
        Returns:
            is_anomalous: Boolean indicating anomaly
            z_scores: Dictionary of z-scores for each feature
        """
        if self.baseline_signature is None or self.baseline_std is None:
            raise ValueError("Baseline not calibrated. Call calibrate_baseline() first.")
        
        features = self.extract_spectral_features(signal_data)
        z_scores = {}
        max_z = 0.0
        
        for key, value in features.items():
            if key in self.baseline_signature and key in self.baseline_std:
                if self.baseline_std[key] > 0:
                    z_score = abs(value - self.baseline_signature[key]) / self.baseline_std[key]
                    z_scores[key] = z_score
                    max_z = max(max_z, z_score)
        
        return max_z > self.detection_threshold, z_scores


class TheftEventSimulator:
    """
    Simulator for generating synthetic theft events with known spectral signatures.
    Used for testing and validation of the fingerprinting engine.
    """
    
    def __init__(self, sampling_rate: float = 1000.0, fundamental_freq: float = 50.0):
        self.sampling_rate = sampling_rate
        self.fundamental_freq = fundamental_freq
    
    def generate_normal_load(self, duration: float = 1.0, load_type: str = 'mixed') -> np.ndarray:
        """
        Generate normal electrical load signal.
        
        Args:
            duration: Signal duration in seconds
            load_type: Type of load ('resistive', 'inductive', 'mixed')
            
        Returns:
            Normal load signal
        """
        t = np.arange(0, duration, 1/self.sampling_rate)
        signal = np.sin(2 * np.pi * self.fundamental_freq * t)
        
        if load_type == 'resistive' or load_type == 'mixed':
            # Add some harmonics typical of resistive loads
            signal += 0.05 * np.sin(2 * np.pi * 3 * self.fundamental_freq * t)  # 3rd harmonic
            signal += 0.02 * np.sin(2 * np.pi * 5 * self.fundamental_freq * t)  # 5th harmonic
        
        if load_type == 'inductive' or load_type == 'mixed':
            # Add phase shift and harmonics typical of inductive loads
            signal += 0.03 * np.sin(2 * np.pi * 2 * self.fundamental_freq * t + np.pi/4)
            signal += 0.01 * np.sin(2 * np.pi * 7 * self.fundamental_freq * t)
        
        # Add small amount of noise
        signal += 0.001 * np.random.randn(len(t))
        
        return signal
    
    def inject_theft_signature(self, 
                              base_signal: np.ndarray, 
                              theft_type: str = 'bypass',
                              severity: float = 0.3) -> np.ndarray:
        """
        Inject theft signature into a normal signal.
        
        Args:
            base_signal: Normal electrical signal
            theft_type: Type of theft ('bypass', 'arcing', 'meter_tampering')
            severity: Severity of theft (0.0 to 1.0)
            
        Returns:
            Signal with theft signature injected
        """
        t = np.arange(0, len(base_signal)/self.sampling_rate, 1/self.sampling_rate)
        theft_signal = np.zeros_like(base_signal)
        
        if theft_type == 'bypass':
            # Bypass theft creates resistive load signature with strong 3rd harmonic
            theft_signal = severity * (
                0.5 * np.sin(2 * np.pi * self.fundamental_freq * t) +
                0.3 * np.sin(2 * np.pi * 3 * self.fundamental_freq * t) +  # Strong 3rd harmonic
                0.1 * np.sin(2 * np.pi * 5 * self.fundamental_freq * t)
            )
        
        elif theft_type == 'arcing':
            # Arcing creates high-frequency noise and inter-harmonics
            theft_signal = severity * (
                0.3 * np.sin(2 * np.pi * self.fundamental_freq * t) +
                0.1 * np.random.randn(len(t)) +  # High-frequency noise
                0.05 * np.sin(2 * np.pi * 150 * t) +  # Inter-harmonic at 150Hz
                0.05 * np.sin(2 * np.pi * 250 * t)    # Inter-harmonic at 250Hz
            )
        
        elif theft_type == 'meter_tampering':
            # Meter tampering creates phase imbalance and harmonic distortion
            theft_signal = severity * (
                0.4 * np.sin(2 * np.pi * self.fundamental_freq * t + 0.2) +  # Phase shift
                0.2 * np.sin(2 * np.pi * 3 * self.fundamental_freq * t) +
                0.1 * np.sin(2 * np.pi * 7 * self.fundamental_freq * t)
            )
        
        return base_signal + theft_signal
    
    def generate_feeder_with_theft(self,
                                  n_meters: int = 10,
                                  theft_meter_indices: List[int] = None,
                                  theft_type: str = 'bypass',
                                  severity: float = 0.3) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Generate a complete feeder scenario with some meters showing theft.
        
        Returns:
            feeder_signal: Aggregate signal at feeder level
            meter_signals: List of individual meter signals
            ground_truth: Dictionary with theft information
        """
        if theft_meter_indices is None:
            theft_meter_indices = []
        
        duration = 1.0  # 1 second of data
        feeder_signal = np.zeros(int(duration * self.sampling_rate))
        meter_signals = []
        
        for i in range(n_meters):
            # Generate normal load for this meter
            meter_signal = self.generate_normal_load(duration, load_type='mixed')
            
            # Inject theft signature if this meter is flagged
            if i in theft_meter_indices:
                meter_signal = self.inject_theft_signature(meter_signal, theft_type, severity)
            
            meter_signals.append(meter_signal)
            feeder_signal += meter_signal
        
        # Add some line losses and noise to feeder signal
        feeder_signal += 0.001 * np.random.randn(len(feeder_signal))
        
        ground_truth = {
            'theft_meters': theft_meter_indices,
            'theft_type': theft_type,
            'severity': severity,
            'n_meters': n_meters
        }
        
        return feeder_signal, meter_signals, ground_truth


if __name__ == "__main__":
    # Demo: Spectral Fingerprinting Engine
    print("=" * 60)
    print("GridPulse AI - Spectral Fingerprinting Engine Demo")
    print("=" * 60)
    
    # Initialize
    fingerprinter = SpectralFingerprinter(sampling_rate=1000.0)
    simulator = TheftEventSimulator(sampling_rate=1000.0)
    
    # Generate test data: Normal scenario
    print("\n1. Testing Normal Operation...")
    feeder_normal, meters_normal, _ = simulator.generate_feeder_with_theft(
        n_meters=5, 
        theft_meter_indices=[]
    )
    
    result_normal = fingerprinter.detect_theft_signature(feeder_normal, meters_normal)
    print(f"   Theft Detected: {result_normal['theft_detected']}")
    print(f"   Confidence: {result_normal['confidence']:.3f}")
    
    # Generate test data: Theft scenario (300Hz noise signature)
    print("\n2. Testing Theft Detection (Bypass at 300Hz)...")
    feeder_theft, meters_theft, ground_truth = simulator.generate_feeder_with_theft(
        n_meters=5,
        theft_meter_indices=[2, 4],  # Meters 2 and 4 are stealing
        theft_type='bypass',
        severity=0.4
    )
    
    result_theft = fingerprinter.detect_theft_signature(feeder_theft, meters_theft)
    print(f"   Theft Detected: {result_theft['theft_detected']}")
    print(f"   Confidence: {result_theft['confidence']:.3f}")
    print(f"   Detected Signatures: {result_theft['theft_signatures']}")
    
    # Show spectral features
    print("\n3. Spectral Features Comparison:")
    print("   Feature | Normal | Theft | Difference")
    print("   " + "-" * 40)
    normal_features = result_normal['feeder_features']
    theft_features = result_theft['feeder_features']
    
    key_features = ['total_harmonic_distortion', 'high_freq_noise_floor', 'spectral_flatness']
    for feature in key_features:
        if feature in normal_features and feature in theft_features:
            diff = theft_features[feature] - normal_features[feature]
            print(f"   {feature}: {normal_features[feature]:.4f} | {theft_features[feature]:.4f} | {diff:+.4f}")
    
    print("\n" + "=" * 60)
    print("Spectral Fingerprinting Engine Ready for Integration")
    print("=" * 60)