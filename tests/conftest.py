"""Pytest fixtures for test suite."""

import pandas as pd
import pytest


@pytest.fixture
def create_time_budget_excel():
    """Factory fixture to create test time budget Excel file."""

    def _create(filepath):
        df = pd.DataFrame(
            {
                "Observation id": ["G213_FAUJea_SDS2_P_M36_V4_25062025"] * 2,
                "Observation date": ["2025-09-06 14:58:19"] * 2,
                "Description": [""] * 2,
                "Time budget start (s)": [0.0, 0.0],
                "Time budget stop (s)": [300.0, 300.0],
                "Time budget duration (s)": [300.0, 300.0],
                "Subject": ["FAUJea", "FAUJea"],
                "Behavior": ["E.D. - Addition", "E.D. - Soustraction"],
                "Modifiers": ["", ""],
                "Total number of occurences": [24, 12],
                "Total duration (s)": [45.5, 23.2],
                "Duration mean (s)": [1.896, 1.933],
                "Duration std dev": [0.5, 0.3],
                "inter-event intervals mean (s)": [168.71, 200.5],
                "inter-event intervals std dev": [126.785, 150.2],
                "% of total length": [15.17, 7.73],
            }
        )
        df.to_excel(filepath, sheet_name="Time budget", index=False)
        return filepath

    return _create


@pytest.fixture
def create_aggregated_excel():
    """Factory fixture to create test aggregated Excel file."""

    def _create(filepath):
        df = pd.DataFrame(
            {
                "Observation id": ["G213_FAUJea_SDS2_P_M36_V4_25062025"] * 3,
                "Observation date": ["2025-09-06 14:58:19"] * 3,
                "Description": [""] * 3,
                "Observation type": ["MEDIA"] * 3,
                "Source": [""] * 3,
                "Time offset (s)": [0.0, 0.0, 0.0],
                "Coding duration": ["00:05:00"] * 3,
                "Media duration (s)": [300.0, 300.0, 300.0],
                "FPS (frame/s)": [30.0, 30.0, 30.0],
                "Subject": ["FAUJea", "FAUJea", "FAUJea"],
                "Observation duration by subject by observation": [300.0, 300.0, 300.0],
                "Behavior": ["E.D. - Addition", "E.D. - Soustraction", "E.D. - Addition"],
                "Behavioral category": ["Eye Direction", "Eye Direction", "Eye Direction"],
                "Behavior type": ["STATE", "STATE", "STATE"],
                "Start (s)": [0.0, 5.5, 15.0],
                "Stop (s)": [5.5, 10.0, 25.0],
                "Duration (s)": [5.5, 4.5, 10.0],
                "Media file name": ["video.mp4"] * 3,
                "Image index start": [0, 165, 450],
                "Image index stop": [165, 300, 750],
                "Image file path start": [""] * 3,
                "Image file path stop": [""] * 3,
                "Comment start": [""] * 3,
                "Comment stop": [""] * 3,
            }
        )
        df.to_excel(filepath, sheet_name="Aggregated events", index=False)
        return filepath

    return _create


@pytest.fixture
def create_boris_files(create_time_budget_excel, create_aggregated_excel):
    """Factory fixture to create both time budget and aggregated files."""

    def _create(directory, filename_base="G213_FAUJea_SDS2_P_M36_V4_25062025"):
        directory.mkdir(parents=True, exist_ok=True)

        time_budget_path = directory / f"{filename_base}.xlsx"
        aggregated_path = directory / f"{filename_base}_agregated.xlsx"

        create_time_budget_excel(time_budget_path)
        create_aggregated_excel(aggregated_path)

        return time_budget_path, aggregated_path

    return _create
