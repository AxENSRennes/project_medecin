"""Filename metadata parsing for Tobii recordings."""

import re
from datetime import datetime
from pathlib import Path


def parse_filename(filename: str) -> dict:
    """Parse Tobii recording filename to extract metadata.

    Filename format: {ID}_{Name}_{Study}_{Group}_{Month}_{Visit}_{Date}
    Example: G213_FAUJea_SDS2_P_M36_V4_25062025

    Args:
        filename: The filename (with or without path and extension)

    Returns:
        dict with keys: id, participant, study, group, month, visit, date, raw_filename
        Returns None values for fields that couldn't be parsed.
    """
    # Extract just the filename without path and extension
    path = Path(filename)
    name = path.stem

    # Remove " Data Export" suffix if present
    name = re.sub(r"\s*Data Export$", "", name)

    # Pattern: ID_Name_Study_Group_Month_Visit_Date
    pattern = r"^([GL]\d+)_([A-Za-z]{6})_([A-Z0-9]+)_([PC])_(M\d+)_(V\d+)_(\d{8})$"
    match = re.match(pattern, name)

    if match:
        id_, participant, study, group, month, visit, date_str = match.groups()

        # Parse date (DDMMYYYY format)
        try:
            date = datetime.strptime(date_str, "%d%m%Y").date()
        except ValueError:
            date = None

        # Parse month number
        month_num = int(month[1:]) if month.startswith("M") else None

        # Parse visit number
        visit_num = int(visit[1:]) if visit.startswith("V") else None

        return {
            "id": id_,
            "participant": participant,
            "study": study,
            "group": "Patient" if group == "P" else "Control",
            "group_code": group,
            "month": month_num,
            "month_code": month,
            "visit": visit_num,
            "visit_code": visit,
            "date": date,
            "date_str": date_str,
            "raw_filename": name,
        }

    # Return dict with None values if pattern doesn't match
    return {
        "id": None,
        "participant": None,
        "study": None,
        "group": None,
        "group_code": None,
        "month": None,
        "month_code": None,
        "visit": None,
        "visit_code": None,
        "date": None,
        "date_str": None,
        "raw_filename": name,
    }
