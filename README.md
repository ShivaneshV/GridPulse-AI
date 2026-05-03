# GridPulse AI

A proof-of-concept for detecting AT&C (Aggregate Technical & Commercial) power losses and meter bypasses using smart meter data. Built for the Karnataka Commerce & Industry hackathon.

## Overview

Traditional theft detection relies on statistical anomalies, which miss subtle bypasses. GridPulse AI takes a different approach by running Fast Fourier Transforms (FFT) on power telemetry to detect the physical harmonic distortion (e.g., 300Hz arcing noise) caused by line tapping.

We also use Temporal Fusion Transformers (TFT) to forecast multi-horizon loads and flag deviations at the transformer level.

## Architecture & Compliance

- **No New Hardware**: Operates entirely on existing smart meter data.
- **DLMS/COSEM**: Data ingestion uses a simulated sidecar API that follows IEC 62056 standards.
- **K-Commerce Integration**: Uses the Unique Business Identifier (UBID) as a primary key. By analyzing spectral load, we can automatically flag if a business registered as "Dormant" is actually running heavy machinery (Theme 1 & 2).

## Local Setup

```bash
# Make setup script executable
chmod +x setup.sh

# Install requirements, generate mock data, and launch dashboard
./setup.sh
```

## Dashboard Features

The Streamlit interface (`src/dashboard/app.py`) includes:
- **Map View**: Real-time plotting of grid anomalies with forensic extraction (FFT signatures).
- **TFT View**: Multi-horizon load forecasting vs actual consumption.
- **Revenue Dashboard**: Tracks estimated AT&C loss reduction and projected ROI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
