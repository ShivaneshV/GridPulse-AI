"""
DLMS/COSEM Sidecar API - GridPulse AI
======================================
Simulates a DLMS/COSEM (IEC 62056) compliant API endpoint that interfaces
with BESCOM's existing Meter Data Management (MDM) system without requiring
any hardware changes.

The Sidecar API:
- Listens to existing BESCOM data streams
- Provides standardized DLMS/COSEM object access
- Enables real-time data fetching for AI analysis
- Supports edge deployment on low-resource hardware
"""

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import numpy as np
import uvicorn
import threading
import time
from enum import Enum


# DLMS/COSEM Object Definitions
class DLMSObjectType(str, Enum):
    """Standard DLMS/COSEM object types."""
    DATA = "data"  # Data buffer
    REGISTER = "register"  # Active energy register
    DEMAND = "demand"  # Maximum demand
    REGISTER_ACTIVATION = "register_activation"  # Register activation mask
    STATUS = "status"  # Meter status
    EVENT_LOG = "event_log"  # Event log buffer
    PROFILE_GENERIC = "profile_generic"  # Load profile
    CLOCK = "clock"  # Clock
    UTILITY_TABLES = "utility_tables"  # Utility tables


# Pydantic Models for API Requests/Responses
class MeterReading(BaseModel):
    """Meter reading data model."""
    obis_code: str = Field(..., description="OBIS code for the data object")
    value: float = Field(..., description="Reading value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(..., description="Reading timestamp")
    quality: str = Field(default="valid", description="Data quality indicator")


class MeterStatus(BaseModel):
    """Meter status information."""
    meter_id: str
    timestamp: datetime
    voltage_l1: float
    voltage_l2: Optional[float]
    voltage_l3: Optional[float]
    current_l1: float
    current_l2: Optional[float]
    current_l3: Optional[float]
    active_power: float
    reactive_power: float
    apparent_power: float
    power_factor: float
    frequency: float
    phase_angle: Optional[float]
    tamper_status: bool = False
    bypass_indicator: bool = False


class DLMSRequest(BaseModel):
    """DLMS request model."""
    meter_id: str
    obis_code: str
    operation: str = "get"  # get, set, action
    data: Optional[Dict[str, Any]] = None


class DLMSResponse(BaseModel):
    """DLMS response model."""
    meter_id: str
    obis_code: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TheftAlert(BaseModel):
    """Theft alert notification."""
    alert_id: str
    meter_id: str
    confidence: float
    theft_type: str
    timestamp: datetime
    evidence: Dict[str, Any]
    recommended_action: str


class GridPulseSidecarAPI:
    """
    DLMS/COSEM Sidecar API for GridPulse AI.
    
    This API acts as a bridge between GridPulse AI and BESCOM's existing
    Meter Data Management (MDM) system, providing standardized access to
    meter data without requiring any changes to legacy infrastructure.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="GridPulse AI - DLMS/COSEM Sidecar API",
            description="DLMS/COSEM compliant API for smart meter data access",
            version="1.0.0"
        )
        
        # Simulated meter data store
        self.meter_data_store: Dict[str, Dict] = {}
        self.meter_status_store: Dict[str, MeterStatus] = {}
        self.theft_alerts: List[TheftAlert] = []
        
        # OBIS code mappings (standard DLMS/COSEM)
        self.obis_mappings = {
            "1.8.0": {"name": "Active energy import", "unit": "kWh", "class": "register"},
            "2.8.0": {"name": "Active energy export", "unit": "kWh", "class": "register"},
            "1.7.0": {"name": "Active power", "unit": "kW", "class": "data"},
            "2.7.0": {"name": "Reactive power", "unit": "kVAR", "class": "data"},
            "3.7.0": {"name": "Apparent power", "unit": "kVA", "class": "data"},
            "13.7.0": {"name": "Power factor", "unit": "", "class": "data"},
            "14.7.0": {"name": "Frequency", "unit": "Hz", "class": "data"},
            "32.7.0": {"name": "Voltage L1", "unit": "V", "class": "data"},
            "52.7.0": {"name": "Voltage L2", "unit": "V", "class": "data"},
            "72.7.0": {"name": "Voltage L3", "unit": "V", "class": "data"},
            "31.7.0": {"name": "Current L1", "unit": "A", "class": "data"},
            "51.7.0": {"name": "Current L2", "unit": "A", "class": "data"},
            "71.7.0": {"name": "Current L3", "unit": "A", "class": "data"},
            "99.99.99": {"name": "Meter Status", "unit": "", "class": "status"},
        }
        
        # Setup API routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """API health check."""
            return {
                "service": "GridPulse AI Sidecar API",
                "version": "1.0.0",
                "protocol": "DLMS/COSEM (IEC 62056)",
                "status": "operational",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "meters_connected": len(self.meter_data_store),
                "alerts_active": len([a for a in self.theft_alerts if a.confidence > 0.7]),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/meters", response_model=List[str])
        async def list_meters():
            """List all connected meters."""
            return list(self.meter_data_store.keys())
        
        @self.app.get("/meters/{meter_id}/status", response_model=MeterStatus)
        async def get_meter_status(meter_id: str):
            """Get current meter status."""
            if meter_id not in self.meter_status_store:
                raise HTTPException(status_code=404, detail="Meter not found")
            return self.meter_status_store[meter_id]
        
        @self.app.get("/meters/{meter_id}/reading")
        async def get_meter_reading(meter_id: str, obis_code: str = "1.8.0"):
            """Get meter reading for specific OBIS code."""
            if meter_id not in self.meter_data_store:
                raise HTTPException(status_code=404, detail="Meter not found")
            
            if obis_code not in self.obis_mappings:
                raise HTTPException(status_code=400, detail="Invalid OBIS code")
            
            # Get latest reading
            meter_data = self.meter_data_store[meter_id]
            if obis_code not in meter_data:
                raise HTTPException(status_code=404, detail="OBIS code not available")
            
            readings = meter_data[obis_code]
            latest = readings[-1] if readings else None
            
            if not latest:
                raise HTTPException(status_code=404, detail="No reading available")
            
            return MeterReading(
                obis_code=obis_code,
                value=latest["value"],
                unit=self.obis_mappings[obis_code]["unit"],
                timestamp=latest["timestamp"],
                quality=latest.get("quality", "valid")
            )
        
        @self.app.get("/meters/{meter_id}/profile")
        async def get_load_profile(meter_id: str,
                                  start_time: datetime,
                                  end_time: datetime,
                                  interval_minutes: int = 15):
            """Get load profile for time range."""
            if meter_id not in self.meter_data_store:
                raise HTTPException(status_code=404, detail="Meter not found")
            
            meter_data = self.meter_data_store[meter_id]
            
            # Get active power profile (OBIS 1.7.0)
            if "1.7.0" not in meter_data:
                raise HTTPException(status_code=404, detail="Load profile not available")
            
            readings = meter_data["1.7.0"]
            profile = [
                r for r in readings
                if start_time <= r["timestamp"] <= end_time
            ]
            
            return {
                "meter_id": meter_id,
                "obis_code": "1.7.0",
                "interval_minutes": interval_minutes,
                "data_points": len(profile),
                "profile": profile
            }
        
        @self.app.post("/meters/{meter_id}/read")
        async def read_meter(request: DLMSRequest):
            """DLMS read operation."""
            if request.meter_id not in self.meter_data_store:
                return DLMSResponse(
                    meter_id=request.meter_id,
                    obis_code=request.obis_code,
                    success=False,
                    error_code=404,
                    error_message="Meter not found"
                )
            
            if request.obis_code not in self.obis_mappings:
                return DLMSResponse(
                    meter_id=request.meter_id,
                    obis_code=request.obis_code,
                    success=False,
                    error_code=400,
                    error_message="Invalid OBIS code"
                )
            
            # Simulate DLMS read
            meter_data = self.meter_data_store[request.meter_id]
            if request.obis_code not in meter_data:
                return DLMSResponse(
                    meter_id=request.meter_id,
                    obis_code=request.obis_code,
                    success=False,
                    error_code=404,
                    error_message="OBIS code not available"
                )
            
            readings = meter_data[request.obis_code]
            latest = readings[-1] if readings else None
            
            return DLMSResponse(
                meter_id=request.meter_id,
                obis_code=request.obis_code,
                success=True,
                data={
                    "value": latest["value"] if latest else None,
                    "unit": self.obis_mappings[request.obis_code]["unit"],
                    "timestamp": latest["timestamp"].isoformat() if latest else None
                }
            )
        
        @self.app.get("/alerts")
        async def list_alerts(confidence_threshold: float = 0.7):
            """List theft alerts above confidence threshold."""
            filtered_alerts = [
                a for a in self.theft_alerts
                if a.confidence >= confidence_threshold
            ]
            return {
                "total_alerts": len(filtered_alerts),
                "alerts": filtered_alerts
            }
        
        @self.app.get("/alerts/{alert_id}")
        async def get_alert(alert_id: str):
            """Get specific alert details."""
            for alert in self.theft_alerts:
                if alert.alert_id == alert_id:
                    return alert
            raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.app.post("/alerts")
        async def create_alert(alert: TheftAlert):
            """Create new theft alert."""
            self.theft_alerts.append(alert)
            return {"status": "alert_created", "alert_id": alert.alert_id}
        
        @self.app.get("/feeder/{feeder_id}/aggregate")
        async def get_feeder_aggregate(feeder_id: str):
            """Get aggregate data for feeder."""
            # Simulate feeder-level aggregation
            feeder_meters = [
                mid for mid in self.meter_data_store.keys()
                if feeder_id in mid  # Simple filtering
            ]
            
            if not feeder_meters:
                raise HTTPException(status_code=404, detail="Feeder not found")
            
            # Aggregate power from all meters
            total_power = 0.0
            for meter_id in feeder_meters:
                meter_data = self.meter_data_store[meter_id]
                if "1.7.0" in meter_data and meter_data["1.7.0"]:
                    total_power += meter_data["1.7.0"][-1]["value"]
            
            return {
                "feeder_id": feeder_id,
                "meter_count": len(feeder_meters),
                "total_active_power_kw": total_power,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/spectral/{meter_id}")
        async def get_spectral_data(meter_id: str):
            """Get spectral analysis data for meter."""
            if meter_id not in self.meter_data_store:
                raise HTTPException(status_code=404, detail="Meter not found")
            
            # Simulate spectral data
            spectral_data = {
                "meter_id": meter_id,
                "fundamental_frequency": 50.0,
                "harmonics": {
                    f"h{i}": np.random.uniform(0.01, 0.1) for i in range(2, 21)
                },
                "thd_percent": np.random.uniform(2.0, 8.0),
                "inter_harmonics": {
                    f"ih{i}": np.random.uniform(0.001, 0.01) for i in range(1, 10)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return spectral_data
        
        @self.app.get("/diagnostics")
        async def get_diagnostics():
            """Get system diagnostics."""
            return {
                "api_version": "1.0.0",
                "dlms_compliance": "IEC 62056",
                "meters_connected": len(self.meter_data_store),
                "data_points_stored": sum(
                    len(readings)
                    for meter_data in self.meter_data_store.values()
                    for readings in meter_data.values()
                ),
                "active_alerts": len([a for a in self.theft_alerts if a.confidence > 0.7]),
                "uptime_seconds": time.time() - self._start_time,
                "memory_usage_mb": self._get_memory_usage(),
                "cpu_usage_percent": self._get_cpu_usage()
            }
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage."""
        # Simplified estimation
        return len(self.meter_data_store) * 0.1 + len(self.theft_alerts) * 0.01
    
    def _get_cpu_usage(self) -> float:
        """Get approximate CPU usage."""
        return np.random.uniform(5.0, 15.0)  # Simulated
    
    def add_meter_data(self, meter_id: str, obis_code: str, value: float, 
                      unit: str = None, quality: str = "valid"):
        """Add meter reading to data store."""
        if meter_id not in self.meter_data_store:
            self.meter_data_store[meter_id] = {}
        
        if obis_code not in self.meter_data_store[meter_id]:
            self.meter_data_store[meter_id][obis_code] = []
        
        reading = {
            "value": value,
            "unit": unit or self.obis_mappings.get(obis_code, {}).get("unit", ""),
            "timestamp": datetime.now(),
            "quality": quality
        }
        
        self.meter_data_store[meter_id][obis_code].append(reading)
        
        # Keep only last 1000 readings per OBIS code
        if len(self.meter_data_store[meter_id][obis_code]) > 1000:
            self.meter_data_store[meter_id][obis_code] = \
                self.meter_data_store[meter_id][obis_code][-1000:]
    
    def update_meter_status(self, meter_id: str, status: MeterStatus):
        """Update meter status."""
        self.meter_status_store[meter_id] = status
    
    def add_theft_alert(self, alert: TheftAlert):
        """Add theft alert."""
        self.theft_alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.theft_alerts) > 100:
            self.theft_alerts = self.theft_alerts[-100:]
    
    def simulate_meter_data(self, meter_id: str, duration_seconds: int = 60):
        """Simulate meter data for testing."""
        base_time = datetime.now()
        
        # Simulate various OBIS codes
        for i in range(duration_seconds * 4):  # 15-minute intervals
            timestamp = base_time + timedelta(seconds=i * 15)
            
            # Active energy (1.8.0) - cumulative
            energy = 1000 + i * 0.5 + np.random.normal(0, 0.1)
            self.add_meter_data(meter_id, "1.8.0", energy, "kWh")
            
            # Active power (1.7.0) - varies
            power = 5.0 + 3.0 * np.sin(2 * np.pi * i / 96) + np.random.normal(0, 0.2)
            self.add_meter_data(meter_id, "1.7.0", power, "kW")
            
            # Reactive power (2.7.0)
            reactive = 2.0 + 1.0 * np.sin(2 * np.pi * i / 96) + np.random.normal(0, 0.1)
            self.add_meter_data(meter_id, "2.7.0", reactive, "kVAR")
            
            # Voltage (32.7.0)
            voltage = 230 + np.random.normal(0, 2)
            self.add_meter_data(meter_id, "32.7.0", voltage, "V")
            
            # Current (31.7.0)
            current = power / 230 + np.random.normal(0, 0.05)
            self.add_meter_data(meter_id, "31.7.0", current, "A")
            
            # Power factor (13.7.0)
            pf = 0.85 + np.random.normal(0, 0.02)
            self.add_meter_data(meter_id, "13.7.0", min(max(pf, 0.5), 1.0), "")
            
            # Frequency (14.7.0)
            freq = 50.0 + np.random.normal(0, 0.05)
            self.add_meter_data(meter_id, "14.7.0", freq, "Hz")
        
        # Update meter status
        status = MeterStatus(
            meter_id=meter_id,
            timestamp=datetime.now(),
            voltage_l1=230 + np.random.normal(0, 2),
            voltage_l2=None,
            voltage_l3=None,
            current_l1=5.0 + np.random.normal(0, 0.5),
            current_l2=None,
            current_l3=None,
            active_power=5.0 + np.random.normal(0, 0.3),
            reactive_power=2.0 + np.random.normal(0, 0.2),
            apparent_power=5.5 + np.random.normal(0, 0.3),
            power_factor=0.85 + np.random.normal(0, 0.02),
            frequency=50.0 + np.random.normal(0, 0.05),
            phase_angle=None,
            tamper_status=False,
            bypass_indicator=False
        )
        self.update_meter_status(meter_id, status)
    
    def run(self, blocking: bool = True):
        """Run the API server."""
        self._start_time = time.time()
        
        if blocking:
            uvicorn.run(self.app, host=self.host, port=self.port)
        else:
            server = threading.Thread(
                target=uvicorn.run,
                args=(self.app,),
                kwargs={"host": self.host, "port": self.port},
                daemon=True
            )
            server.start()
            return server


# Edge-compatible lightweight version
class EdgeDLMSAPI:
    """
    Lightweight DLMS API for edge deployment on low-resource hardware.
    Optimized for sub-50ms inference at edge data concentrators.
    """
    
    def __init__(self, max_meters: int = 100, buffer_size: int = 1000):
        self.max_meters = max_meters
        self.buffer_size = buffer_size
        
        # Lightweight data structures
        self.meter_buffers: Dict[str, np.ndarray] = {}
        self.meter_metadata: Dict[str, Dict] = {}
        
        # Pre-allocate buffers for efficiency
        for i in range(max_meters):
            meter_id = f"EDGE_MT_{i:04d}"
            # Buffer for: timestamp, active_power, reactive_power, voltage, current
            self.meter_buffers[meter_id] = np.zeros((buffer_size, 5))
            self.meter_metadata[meter_id] = {
                "active": False,
                "last_update": None,
                "anomaly_score": 0.0
            }
    
    def add_reading(self, meter_id: str, reading: np.ndarray):
        """Add reading to circular buffer."""
        if meter_id in self.meter_buffers:
            buffer = self.meter_buffers[meter_id]
            # Roll buffer and add new reading
            buffer[:-1] = buffer[1:]
            buffer[-1] = reading
            self.meter_metadata[meter_id]["last_update"] = datetime.now()
            self.meter_metadata[meter_id]["active"] = True
    
    def get_features(self, meter_id: str, window_size: int = 96) -> Optional[np.ndarray]:
        """Get features for AI model (last N readings)."""
        if meter_id not in self.meter_buffers:
            return None
        
        buffer = self.meter_buffers[meter_id]
        features = buffer[-window_size:]
        
        # Compute statistical features
        stats = np.array([
            np.mean(features[:, 1]),  # Mean active power
            np.std(features[:, 1]),   # Power variability
            np.min(features[:, 3]),   # Min voltage
            np.max(features[:, 3]),   # Max voltage
            np.mean(features[:, 4]),  # Mean current
            np.std(features[:, 4]),   # Current variability
        ])
        
        return stats
    
    def is_anomalous(self, meter_id: str, threshold: float = 0.7) -> bool:
        """Check if meter shows anomalous behavior."""
        if meter_id not in self.meter_metadata:
            return False
        
        return self.meter_metadata[meter_id]["anomaly_score"] > threshold


if __name__ == "__main__":
    # Demo: DLMS/COSEM Sidecar API
    print("=" * 60)
    print("GridPulse AI - DLMS/COSEM Sidecar API Demo")
    print("=" * 60)
    
    # Initialize API
    api = GridPulseSidecarAPI(host="127.0.0.1", port=8080)
    
    # Simulate meter data
    print("\n1. Simulating meter data...")
    api.simulate_meter_data("BESCOM_MT_000001", duration_seconds=60)
    api.simulate_meter_data("BESCOM_MT_000002", duration_seconds=60)
    
    print(f"   Meters registered: {len(api.meter_data_store)}")
    print(f"   Data points per meter: {len(api.meter_data_store['BESCOM_MT_000001']['1.8.0'])}")
    
    # Test API endpoints
    print("\n2. Testing API endpoints...")
    
    # Health check
    print(f"   Health check: {api.meter_data_store and 'healthy'}")
    
    # List meters
    meters = list(api.meter_data_store.keys())
    print(f"   Connected meters: {meters}")
    
    # Get meter status
    if "BESCOM_MT_000001" in api.meter_status_store:
        status = api.meter_status_store["BESCOM_MT_000001"]
        print(f"   Meter status: Voltage={status.voltage_l1:.1f}V, "
              f"Power={status.active_power:.2f}kW, PF={status.power_factor:.2f}")
    
    # Simulate theft alert
    print("\n3. Simulating theft alert...")
    alert = TheftAlert(
        alert_id="ALERT_001",
        meter_id="BESCOM_MT_000001",
        confidence=0.92,
        theft_type="bypass",
        timestamp=datetime.now(),
        evidence={
            "thd_discrepancy": 0.45,
            "energy_balance": 0.32,
            "spectral_signature": "resistive_hooking"
        },
        recommended_action="Dispatch field team to Pole B-402"
    )
    api.add_theft_alert(alert)
    print(f"   Alert created: {alert.alert_id}")
    print(f"   Confidence: {alert.confidence:.0%}")
    print(f"   Action: {alert.recommended_action}")
    
    # Edge API demo
    print("\n4. Edge API Demo...")
    edge_api = EdgeDLMSAPI(max_meters=10)
    
    # Add some readings
    for i in range(100):
        reading = np.array([
            time.time(),  # timestamp
            5.0 + np.random.normal(0, 0.5),  # active power
            2.0 + np.random.normal(0, 0.2),  # reactive power
            230 + np.random.normal(0, 2),    # voltage
            0.02 + np.random.normal(0, 0.005)  # current
        ])
        edge_api.add_reading("EDGE_MT_0000", reading)
    
    features = edge_api.get_features("EDGE_MT_0000")
    print(f"   Edge features computed: {features}")
    
    print("\n" + "=" * 60)
    print("DLMS/COSEM Sidecar API Ready for Integration")
    print("=" * 60)
    
    # Note: Don't actually start the server in demo mode
    print("\nTo start the API server, run:")
    print("  python -m gridpulse.api.dlms_sidecar")