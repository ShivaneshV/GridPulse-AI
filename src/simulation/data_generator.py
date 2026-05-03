"""
Synthetic Data Generator - GridPulse AI
========================================
Generates realistic synthetic smart meter data with known theft patterns
for training and validation of the GridPulse AI system.

Creates:
- Normal consumption patterns with seasonality and weather effects
- Theft events with various signatures (bypass, arcing, meter tampering)
- Grid topology with transformers, feeders, and meters
- Data gaps and noise to simulate real-world conditions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os


class GridTopology:
    """
    Represents the electrical grid topology with transformers, feeders, and meters.
    """
    
    def __init__(self, 
                 n_transformers: int = 5,
                 n_feeders_per_transformer: int = 3,
                 n_meters_per_feeder: int = 10,
                 area_bounds: Tuple = (12.9, 77.5, 13.1, 77.7)):  # Bengaluru bounds
        """
        Initialize grid topology.
        
        Args:
            n_transformers: Number of distribution transformers
            n_feeders_per_transformer: Average feeders per transformer
            n_meters_per_feeder: Average meters per feeder
            area_bounds: (min_lat, min_lon, max_lat, max_lon) for meter locations
        """
        self.n_transformers = n_transformers
        self.n_feeders_per_transformer = n_feeders_per_transformer
        self.n_meters_per_feeder = n_meters_per_feeder
        self.area_bounds = area_bounds
        
        # Generate topology
        self.transformers = self._generate_transformers()
        self.feeders = self._generate_feeders()
        self.meters = self._generate_meters()
        
        # Build relationships
        self.transformer_to_feeders = {}
        self.feeder_to_meters = {}
        self.meter_to_feeder = {}
        self.feeder_to_transformer = {}
        self._build_relationships()
    
    def _generate_transformers(self) -> List[Dict]:
        """Generate distribution transformers with random locations."""
        transformers = []
        for i in range(self.n_transformers):
            transformer = {
                'id': f'TR_{i+1:03d}',
                'latitude': np.random.uniform(self.area_bounds[0], self.area_bounds[2]),
                'longitude': np.random.uniform(self.area_bounds[1], self.area_bounds[3]),
                'capacity_kva': np.random.choice([100, 160, 200, 250, 315, 400, 500]),
                'age_years': np.random.uniform(1, 30),
                'type': np.random.choice(['pole_mounted', 'ground_mounted', 'indoor']),
                'voltage_level': np.random.choice(['11kV', '22kV'])
            }
            transformers.append(transformer)
        return transformers
    
    def _generate_feeders(self) -> List[Dict]:
        """Generate feeders connecting transformers to meters."""
        feeders = []
        feeder_id = 0
        for tr in self.transformers:
            n_feeders = np.random.randint(1, self.n_feeders_per_transformer + 1)
            for j in range(n_feeders):
                feeder = {
                    'id': f'FD_{feeder_id+1:04d}',
                    'transformer_id': tr['id'],
                    'phase': np.random.choice(['R', 'Y', 'B', 'RY', 'YB', 'BR', 'RYB']),
                    'length_km': np.random.uniform(0.1, 2.0),
                    'cable_type': np.random.choice(['ABC', 'AAC', 'ACSR']),
                    'age_years': np.random.uniform(1, 25)
                }
                feeders.append(feeder)
                feeder_id += 1
        return feeders
    
    def _generate_meters(self) -> List[Dict]:
        """Generate smart meters with locations and consumer types."""
        meters = []
        meter_id = 0
        for feeder in self.feeders:
            n_meters = np.random.randint(5, self.n_meters_per_feeder + 5)
            for k in range(n_meters):
                # Generate meter location near feeder
                base_tr = next(t for t in self.transformers if t['id'] == feeder['transformer_id'])
                meter = {
                    'id': f'MT_{meter_id+1:06d}',
                    'feeder_id': feeder['id'],
                    'transformer_id': feeder['transformer_id'],
                    'latitude': base_tr['latitude'] + np.random.uniform(-0.01, 0.01),
                    'longitude': base_tr['longitude'] + np.random.uniform(-0.01, 0.01),
                    'consumer_type': np.random.choice([
                        'residential', 'commercial', 'industrial', 'agricultural', 'government'
                    ], p=[0.5, 0.2, 0.1, 0.15, 0.05]),
                    'sanctioned_load_kw': self._get_sanctioned_load_kw(),
                    'connection_type': np.random.choice(['LT', 'HT']),
                    'tariff_category': np.random.choice(['domestic', 'commercial', 'industrial', 'agricultural']),
                    'installation_date': datetime.now() - timedelta(days=np.random.randint(30, 3650))
                }
                meters.append(meter)
                meter_id += 1
        return meters
    
    def _get_sanctioned_load_kw(self) -> float:
        """Get sanctioned load based on consumer type."""
        consumer_loads = {
            'residential': (2, 10),
            'commercial': (5, 50),
            'industrial': (20, 200),
            'agricultural': (5, 30),
            'government': (10, 100)
        }
        # This will be called with a consumer_type, but for simplicity we'll return a random range
        low, high = (2, 50)  # Default range
        return np.random.uniform(low, high)
    
    def _build_relationships(self):
        """Build bidirectional relationships between grid elements."""
        for feeder in self.feeders:
            tr_id = feeder['transformer_id']
            if tr_id not in self.transformer_to_feeders:
                self.transformer_to_feeders[tr_id] = []
            self.transformer_to_feeders[tr_id].append(feeder['id'])
            self.feeder_to_transformer[feeder['id']] = tr_id
        
        for meter in self.meters:
            fd_id = meter['feeder_id']
            if fd_id not in self.feeder_to_meters:
                self.feeder_to_meters[fd_id] = []
            self.feeder_to_meters[fd_id].append(meter['id'])
            self.meter_to_feeder[meter['id']] = fd_id
    
    def to_dict(self) -> Dict:
        """Convert topology to dictionary."""
        return {
            'transformers': self.transformers,
            'feeders': self.feeders,
            'meters': self.meters,
            'relationships': {
                'transformer_to_feeders': self.transformer_to_feeders,
                'feeder_to_meters': self.feeder_to_meters,
                'meter_to_feeder': self.meter_to_feeder,
                'feeder_to_transformer': self.feeder_to_transformer
            }
        }


class ConsumptionPatternGenerator:
    """
    Generates realistic electricity consumption patterns for different consumer types.
    """
    
    def __init__(self, weather_api_data: Optional[Dict] = None):
        """
        Initialize consumption pattern generator.
        
        Args:
            weather_api_data: Optional weather data for Karnataka region
        """
        self.weather_data = weather_api_data or self._generate_synthetic_weather()
        
        # Base load profiles for different consumer types (24-hour patterns)
        self.base_profiles = {
            'residential': self._residential_profile(),
            'commercial': self._commercial_profile(),
            'industrial': self._industrial_profile(),
            'agricultural': self._agricultural_profile(),
            'government': self._government_profile()
        }
        
        # Seasonal adjustment factors
        self.seasonal_factors = {
            'summer': 1.3,    # Higher AC usage
            'monsoon': 0.9,   # Lower usage
            'winter': 0.8,    # Lower usage
            'festival': 1.2   # Festival season boost
        }
    
    def _residential_profile(self) -> np.ndarray:
        """Residential consumption pattern (peak morning and evening)."""
        profile = np.zeros(24)
        # Morning peak (6-9 AM)
        profile[6:9] = [0.6, 0.8, 0.7]
        # Day low (9 AM - 5 PM)
        profile[9:17] = 0.4
        # Evening peak (5-10 PM)
        profile[17:22] = [0.6, 0.8, 1.0, 0.9, 0.7]
        # Night low (10 PM - 6 AM)
        profile[22:] = 0.2
        profile[:6] = 0.2
        return profile / profile.max()  # Normalize
    
    def _commercial_profile(self) -> np.ndarray:
        """Commercial consumption pattern (peak during business hours)."""
        profile = np.zeros(24)
        # Business hours (8 AM - 8 PM)
        profile[8:20] = 0.9
        profile[10:18] = 1.0  # Peak business hours
        # Off hours
        profile[:8] = 0.3
        profile[20:] = 0.3
        return profile / profile.max()
    
    def _industrial_profile(self) -> np.ndarray:
        """Industrial consumption pattern (24/7 with shifts)."""
        profile = np.zeros(24)
        # Three shifts
        profile[6:14] = 0.8  # Morning shift
        profile[14:22] = 1.0  # Evening shift (peak)
        profile[22:] = 0.6   # Night shift
        profile[:6] = 0.6
        return profile / profile.max()
    
    def _agricultural_profile(self) -> np.ndarray:
        """Agricultural consumption pattern (pump operation hours)."""
        profile = np.zeros(24)
        # Pump operation typically during off-peak hours
        profile[5:8] = 0.9   # Early morning
        profile[18:21] = 0.9  # Evening
        # Off hours
        profile[8:18] = 0.2
        profile[21:] = 0.1
        profile[:5] = 0.1
        return profile / profile.max()
    
    def _government_profile(self) -> np.ndarray:
        """Government consumption pattern (office hours)."""
        profile = np.zeros(24)
        # Office hours (9 AM - 6 PM)
        profile[9:18] = 0.8
        profile[11:15] = 1.0  # Peak hours
        # Off hours
        profile[:9] = 0.2
        profile[18:] = 0.2
        return profile / profile.max()
    
    def _generate_synthetic_weather(self) -> pd.DataFrame:
        """Generate synthetic weather data for Karnataka."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='h')
        
        weather_data = pd.DataFrame(index=dates)
        
        # Temperature (seasonal pattern)
        base_temp = 28  # Base temperature
        day_of_year = weather_data.index.dayofyear
        weather_data['temperature'] = base_temp + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, len(dates))
        
        # Humidity (higher during monsoon)
        monsoon_mask = (weather_data.index.month >= 6) & (weather_data.index.month <= 9)
        weather_data['humidity'] = np.where(monsoon_mask, 
                                           np.random.normal(80, 10, len(dates)), 
                                           np.random.normal(50, 15, len(dates)))
        
        # Is_holiday (weekends + holidays)
        weather_data['is_weekend'] = weather_data.index.weekday.isin([5, 6]).astype(int)
        
        # Is_festival (random festival days)
        festival_days = np.random.choice(365, size=20, replace=False)
        weather_data['is_festival'] = day_of_year.isin(festival_days).astype(int)
        
        return weather_data
    
    def generate_consumption(self,
                           meter: Dict,
                           start_date: datetime,
                           end_date: datetime,
                           interval_minutes: int = 15,
                           noise_level: float = 0.05) -> pd.DataFrame:
        """
        Generate consumption data for a meter.
        
        Args:
            meter: Meter metadata
            start_date: Start date
            end_date: End date
            interval_minutes: Data interval in minutes
            noise_level: Noise level as fraction of signal
            
        Returns:
            DataFrame with timestamp and consumption values
        """
        # Create time index
        date_range = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
        
        # Get base profile for consumer type
        consumer_type = meter.get('consumer_type', 'residential')
        base_profile = self.base_profiles.get(consumer_type, self.base_profiles['residential'])
        
        # Get sanctioned load
        sanctioned_load = meter.get('sanctioned_load_kw', 5.0)
        
        # Generate base consumption
        n_points = len(date_range)
        consumption = np.zeros(n_points)
        
        for i, ts in enumerate(date_range):
            hour = ts.hour + ts.minute / 60.0
            hour_idx = int(hour) % 24
            
            # Base profile value
            profile_value = base_profile[hour_idx]
            
            # Seasonal adjustment
            month = ts.month
            if month in [3, 4, 5]:
                season = 'summer'
            elif month in [6, 7, 8, 9]:
                season = 'monsoon'
            elif month in [10, 11]:
                season = 'festival'
            else:
                season = 'winter'
            
            seasonal_factor = self.seasonal_factors.get(season, 1.0)
            
            # Temperature effect (higher temp = more AC usage)
            temp_idx = int((ts - date_range[0]).total_seconds() / 3600)
            if temp_idx < len(self.weather_data):
                temp = self.weather_data.iloc[temp_idx]['temperature']
                temp_effect = 1.0 + 0.02 * max(0, temp - 30)  # 2% increase per degree above 30°C
            else:
                temp_effect = 1.0
            
            # Weekend effect
            is_weekend = ts.weekday() in [5, 6]
            weekend_factor = 1.1 if consumer_type == 'residential' and is_weekend else 1.0
            
            # Calculate final consumption
            consumption[i] = (
                sanctioned_load * 
                profile_value * 
                seasonal_factor * 
                temp_effect * 
                weekend_factor
            )
        
        # Add noise
        consumption += np.random.normal(0, noise_level * sanctioned_load, n_points)
        
        # Ensure non-negative
        consumption = np.maximum(consumption, 0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'meter_id': meter['id'],
            'consumption_kw': consumption,
            'voltage_v': 230 + np.random.normal(0, 5, n_points),  # 230V nominal
            'current_a': consumption / 230 + np.random.normal(0, 0.1, n_points),
            'power_factor': 0.85 + np.random.normal(0, 0.05, n_points),
            'frequency_hz': 50.0 + np.random.normal(0, 0.1, n_points)
        })
        
        # Clamp values to realistic ranges
        df['power_factor'] = df['power_factor'].clip(0.5, 1.0)
        df['frequency_hz'] = df['frequency_hz'].clip(49.5, 50.5)
        df['voltage_v'] = df['voltage_v'].clip(200, 260)
        
        return df


class TheftEventGenerator:
    """
    Generates theft events with various signatures for training and testing.
    """
    
    def __init__(self):
        self.theft_types = ['bypass', 'arcing', 'meter_tampering', 'phase_imbalance']
    
    def generate_theft_event(self,
                           consumption_df: pd.DataFrame,
                           theft_type: str = 'bypass',
                           start_time: Optional[datetime] = None,
                           duration_hours: float = 4.0,
                           severity: float = 0.3) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject theft signature into consumption data.
        
        Args:
            consumption_df: Normal consumption data
            theft_type: Type of theft to simulate
            start_time: When theft starts (None for random)
            duration_hours: Duration of theft in hours
            severity: Severity of theft (0.0 to 1.0)
            
        Returns:
            Modified consumption DataFrame and theft metadata
        """
        df = consumption_df.copy()
        
        # Determine theft start time
        if start_time is None:
            # Random start within the data range
            n_points = len(df)
            duration_points = int(duration_hours * 4)  # 15-min intervals
            start_idx = np.random.randint(0, max(1, n_points - duration_points))
        else:
            start_idx = df.index[df['timestamp'] >= start_time][0]
            duration_points = int(duration_hours * 4)
        
        end_idx = min(start_idx + duration_points, len(df))
        
        # Apply theft signature
        theft_signature = self._apply_theft_signature(
            df.iloc[start_idx:end_idx],
            theft_type,
            severity
        )
        
        df.iloc[start_idx:end_idx] = theft_signature
        
        # Create theft metadata
        theft_metadata = {
            'theft_type': theft_type,
            'start_time': df.iloc[start_idx]['timestamp'],
            'end_time': df.iloc[end_idx - 1]['timestamp'],
            'duration_hours': duration_hours,
            'severity': severity,
            'energy_stolen_kwh': self._calculate_stolen_energy(
                consumption_df.iloc[start_idx:end_idx],
                df.iloc[start_idx:end_idx],
                theft_type
            ),
            'spectral_signature': self._get_spectral_signature(theft_type)
        }
        
        return df, theft_metadata
    
    def _apply_theft_signature(self, 
                              df_chunk: pd.DataFrame, 
                              theft_type: str, 
                              severity: float) -> pd.DataFrame:
        """Apply specific theft signature to data chunk."""
        df = df_chunk.copy()
        
        if theft_type == 'bypass':
            # Bypass: Reduced consumption with harmonic distortion
            df['consumption_kw'] *= (1 - severity * 0.7)  # Reduce consumption
            df['current_a'] *= (1 - severity * 0.6)
            # Add harmonic distortion indicators
            df['power_factor'] *= (1 + severity * 0.1)  # Slightly higher PF
            df['total_harmonic_distortion'] = severity * 15  # THD percentage
        
        elif theft_type == 'arcing':
            # Arcing: Intermittent consumption with noise
            noise_factor = severity * 0.5
            df['consumption_kw'] *= (1 + np.random.normal(0, noise_factor, len(df)))
            df['voltage_v'] += np.random.normal(0, severity * 10, len(df))
            df['current_a'] *= (1 + np.random.normal(0, noise_factor, len(df)))
            df['power_factor'] *= (1 - severity * 0.2)  # Lower PF due to arcing
        
        elif theft_type == 'meter_tampering':
            # Meter tampering: Sudden drop in consumption
            df['consumption_kw'] *= (1 - severity * 0.8)
            df['current_a'] *= (1 - severity * 0.7)
            # Add phase imbalance
            df['voltage_v'] *= (1 + np.random.normal(0, severity * 0.05, len(df)))
        
        elif theft_type == 'phase_imbalance':
            # Phase imbalance: Unbalanced voltage and current
            df['voltage_v'] *= (1 + np.random.choice([-1, 1], len(df)) * severity * 0.1)
            df['current_a'] *= (1 + np.random.choice([-1, 1], len(df)) * severity * 0.15)
        
        return df
    
    def _calculate_stolen_energy(self, 
                                normal_df: pd.DataFrame, 
                                theft_df: pd.DataFrame, 
                                theft_type: str) -> float:
        """Calculate energy stolen during theft event."""
        normal_energy = normal_df['consumption_kw'].sum() / 4  # 15-min intervals
        theft_energy = theft_df['consumption_kw'].sum() / 4
        return max(0, normal_energy - theft_energy)
    
    def _get_spectral_signature(self, theft_type: str) -> Dict[str, float]:
        """Get expected spectral signature for theft type."""
        signatures = {
            'bypass': {
                'third_harmonic': 0.3,
                'fifth_harmonic': 0.1,
                'thd': 15.0,
                'power_factor': 0.95
            },
            'arcing': {
                'high_freq_noise': 0.5,
                'inter_harmonics': 0.2,
                'thd': 25.0,
                'power_factor': 0.7
            },
            'meter_tampering': {
                'phase_shift': 0.2,
                'third_harmonic': 0.2,
                'thd': 10.0,
                'power_factor': 0.8
            },
            'phase_imbalance': {
                'negative_sequence': 0.15,
                'zero_sequence': 0.1,
                'thd': 8.0,
                'power_factor': 0.85
            }
        }
        return signatures.get(theft_type, signatures['bypass'])


class SyntheticDatasetGenerator:
    """
    Main class for generating complete synthetic datasets for GridPulse AI.
    """
    
    def __init__(self, 
                 grid_topology: Optional[GridTopology] = None,
                 weather_data: Optional[pd.DataFrame] = None):
        """
        Initialize synthetic dataset generator.
        
        Args:
            grid_topology: Pre-built grid topology (None to generate new)
            weather_data: Pre-built weather data (None to generate new)
        """
        self.topology = grid_topology or GridTopology()
        self.consumption_gen = ConsumptionPatternGenerator(weather_data)
        self.theft_gen = TheftEventGenerator()
        
        # Storage for generated data
        self.meter_data = {}
        self.theft_events = {}
        self.feeder_aggregates = {}
    
    def generate_dataset(self,
                        start_date: datetime,
                        end_date: datetime,
                        theft_percentage: float = 0.1,
                        include_data_gaps: bool = True,
                        gap_percentage: float = 0.02) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Args:
            start_date: Dataset start date
            end_date: Dataset end date
            theft_percentage: Fraction of meters with theft events
            include_data_gaps: Whether to include data gaps
            gap_percentage: Fraction of data points to remove as gaps
            
        Returns:
            Dictionary with 'meters', 'feeders', 'theft_events', 'topology'
        """
        print(f"Generating synthetic dataset from {start_date} to {end_date}...")
        
        # Generate consumption for all meters
        self.meter_data = {}
        for meter in self.topology.meters:
            meter_df = self.consumption_gen.generate_consumption(
                meter, start_date, end_date
            )
            self.meter_data[meter['id']] = meter_df
        
        # Select meters for theft events
        n_theft_meters = max(1, int(len(self.topology.meters) * theft_percentage))
        theft_meter_ids = np.random.choice(
            [m['id'] for m in self.topology.meters],
            size=n_theft_meters,
            replace=False
        )
        
        # Generate theft events
        self.theft_events = {}
        for meter_id in theft_meter_ids:
            meter_df = self.meter_data[meter_id]
            
            # Random theft parameters
            theft_type = np.random.choice(self.theft_gen.theft_types)
            severity = np.random.uniform(0.2, 0.6)
            duration = np.random.uniform(2, 8)  # hours
            
            # Generate theft event
            modified_df, theft_meta = self.theft_gen.generate_theft_event(
                meter_df,
                theft_type=theft_type,
                duration_hours=duration,
                severity=severity
            )
            
            self.meter_data[meter_id] = modified_df
            self.theft_events[meter_id] = theft_meta
        
        # Add data gaps
        if include_data_gaps:
            self._add_data_gaps(gap_percentage)
        
        # Generate feeder aggregates
        self._generate_feeder_aggregates()
        
        print(f"Dataset generated: {len(self.meter_data)} meters, {len(self.theft_events)} theft events")
        
        return {
            'meters': self.meter_data,
            'feeders': self.feeder_aggregates,
            'theft_events': self.theft_events,
            'topology': self.topology.to_dict()
        }
    
    def _add_data_gaps(self, gap_percentage: float):
        """Add random data gaps to simulate communication failures."""
        for meter_id, df in self.meter_data.items():
            n_gaps = int(len(df) * gap_percentage)
            gap_indices = np.random.choice(len(df), size=n_gaps, replace=False)
            
            # Set gaps as NaN
            self.meter_data[meter_id].iloc[gap_indices, self.meter_data[meter_id].columns.get_loc('consumption_kw')] = np.nan
    
    def _generate_feeder_aggregates(self):
        """Generate feeder-level aggregate data."""
        self.feeder_aggregates = {}
        
        for feeder_id, meter_ids in self.topology.feeder_to_meters.items():
            # Aggregate all meters in this feeder
            feeder_data = None
            
            for meter_id in meter_ids:
                if meter_id in self.meter_data:
                    meter_df = self.meter_data[meter_id][['timestamp', 'consumption_kw']].copy()
                    meter_df.columns = ['timestamp', f'{meter_id}_consumption']
                    
                    if feeder_data is None:
                        feeder_data = meter_df
                    else:
                        feeder_data = feeder_data.merge(meter_df, on='timestamp', how='outer')
            
            if feeder_data is not None:
                # Sum all meter consumptions
                consumption_cols = [col for col in feeder_data.columns if col.endswith('_consumption')]
                feeder_data['total_consumption_kw'] = feeder_data[consumption_cols].sum(axis=1)
                feeder_data['feeder_id'] = feeder_id
                
                self.feeder_aggregates[feeder_id] = feeder_data
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, datetime) or isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, np.bool_):
                return bool(obj)
            # Catch any other numpy types like numpy.str_
            if type(obj).__module__ == 'numpy':
                try:
                    return obj.item()
                except Exception:
                    return str(obj)
            return super().default(obj)

    def save_dataset(self, output_dir: str):
        """Save dataset to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save meter data
        meters_dir = os.path.join(output_dir, 'meters')
        os.makedirs(meters_dir, exist_ok=True)
        
        for meter_id, df in self.meter_data.items():
            df.to_csv(os.path.join(meters_dir, f'{meter_id}.csv'), index=False)
        
        # Save feeder aggregates
        feeders_dir = os.path.join(output_dir, 'feeders')
        os.makedirs(feeders_dir, exist_ok=True)
        
        for feeder_id, df in self.feeder_aggregates.items():
            df.to_csv(os.path.join(feeders_dir, f'{feeder_id}.csv'), index=False)
        
        # Save theft events
        with open(os.path.join(output_dir, 'theft_events.json'), 'w') as f:
            # Convert datetime objects to strings
            theft_events_serializable = {}
            for meter_id, event in self.theft_events.items():
                event_copy = event.copy()
                event_copy['start_time'] = event_copy['start_time'].isoformat()
                event_copy['end_time'] = event_copy['end_time'].isoformat()
                theft_events_serializable[meter_id] = event_copy
            json.dump(theft_events_serializable, f, indent=2, cls=self.NpEncoder)
        
        # Save topology
        with open(os.path.join(output_dir, 'topology.json'), 'w') as f:
            json.dump(self.topology.to_dict(), f, indent=2, cls=self.NpEncoder)
        
        print(f"Dataset saved to {output_dir}")
    
    def load_dataset(self, input_dir: str):
        """Load dataset from files."""
        # Load meter data
        meters_dir = os.path.join(input_dir, 'meters')
        self.meter_data = {}
        
        for filename in os.listdir(meters_dir):
            if filename.endswith('.csv'):
                meter_id = filename.replace('.csv', '')
                self.meter_data[meter_id] = pd.read_csv(os.path.join(meters_dir, filename))
        
        # Load feeder aggregates
        feeders_dir = os.path.join(input_dir, 'feeders')
        self.feeder_aggregates = {}
        
        for filename in os.listdir(feeders_dir):
            if filename.endswith('.csv'):
                feeder_id = filename.replace('.csv', '')
                self.feeder_aggregates[feeder_id] = pd.read_csv(os.path.join(feeders_dir, filename))
        
        # Load theft events
        with open(os.path.join(input_dir, 'theft_events.json'), 'r') as f:
            theft_events_serializable = json.load(f)
            self.theft_events = {}
            for meter_id, event in theft_events_serializable.items():
                event_copy = event.copy()
                event_copy['start_time'] = datetime.fromisoformat(event_copy['start_time'])
                event_copy['end_time'] = datetime.fromisoformat(event_copy['end_time'])
                self.theft_events[meter_id] = event_copy
        
        # Load topology
        with open(os.path.join(input_dir, 'topology.json'), 'r') as f:
            topology_dict = json.load(f)
            # Reconstruct topology object (simplified)
            self.topology = GridTopology()
        
        print(f"Dataset loaded from {input_dir}")


if __name__ == "__main__":
    # Demo: Synthetic Dataset Generation
    print("=" * 60)
    print("GridPulse AI - Synthetic Dataset Generator Demo")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticDatasetGenerator()
    
    # Generate dataset for 1 week
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)
    
    dataset = generator.generate_dataset(
        start_date=start_date,
        end_date=end_date,
        theft_percentage=0.15,  # 15% meters have theft
        include_data_gaps=True,
        gap_percentage=0.01
    )
    
    print(f"\n1. Grid Topology:")
    print(f"   Transformers: {len(generator.topology.transformers)}")
    print(f"   Feeders: {len(generator.topology.feeders)}")
    print(f"   Meters: {len(generator.topology.meters)}")
    
    print(f"\n2. Generated Data:")
    print(f"   Meter records: {sum(len(df) for df in dataset['meters'].values())}")
    print(f"   Feeder records: {sum(len(df) for df in dataset['feeders'].values())}")
    print(f"   Theft events: {len(dataset['theft_events'])}")
    
    print(f"\n3. Theft Event Summary:")
    for meter_id, event in list(dataset['theft_events'].items())[:3]:
        print(f"   Meter {meter_id}: {event['theft_type']} theft, "
              f"severity={event['severity']:.2f}, "
              f"energy_stolen={event['energy_stolen_kwh']:.1f} kWh")
    
    print(f"\n4. Sample Meter Data:")
    sample_meter = list(dataset['meters'].keys())[0]
    sample_df = dataset['meters'][sample_meter]
    print(f"   Meter: {sample_meter}")
    print(f"   Shape: {sample_df.shape}")
    print(f"   Columns: {list(sample_df.columns)}")
    print(f"   Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
    print(f"   Avg consumption: {sample_df['consumption_kw'].mean():.2f} kW")
    print(f"   Data gaps: {sample_df['consumption_kw'].isna().sum()}")
    
    # Save dataset
    output_dir = "data/synthetic/demo_dataset"
    generator.save_dataset(output_dir)
    
    print("\n" + "=" * 60)
    print("Synthetic Dataset Generator Ready for Integration")
    print("=" * 60)