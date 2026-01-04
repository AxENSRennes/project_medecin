# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data repository for eye-tracking research (SDS2 study). It contains no source code - only raw and aggregated data files from eye-tracking experiments.

## Data Structure

```
Data/
├── data_G/          # G-series recordings
│   ├── Boris/       # BORIS behavioral observation data (.xlsx)
│   └── Tobii/       # Tobii eye-tracker exports (.tsv)
└── data_L/          # L-series recordings
    ├── Boris/       # BORIS behavioral observation data (.xlsx)
    └── Tobii/       # Tobii eye-tracker exports (.tsv)
```

## File Naming Convention

Files follow the pattern: `{ID}_{Name}_{Study}_{Group}_{Month}_{Visit}_{Date}`

Example: `G213_FAUJea_SDS2_P_M36_V4_25062025`

- **ID**: Recording identifier (G### for data_G, L### for data_L)
- **Name**: 6-character participant code (e.g., FAUJea, BENNaw)
- **Study**: Study identifier (SDS2)
- **Group**: P (Patient) or C (Control)
- **Month**: Timepoint - M0, M12, M24, or M36
- **Visit**: Visit number - V1, V2, V3, or V4
- **Date**: Recording date in DDMMYYYY format

## Data Sources

- **Tobii (.tsv)**: Raw eye-tracking data including gaze points, pupil diameter, gaze direction, gyroscope, and accelerometer readings
- **Boris (.xlsx)**: Behavioral observation data with `_agregated` variants containing processed summaries
