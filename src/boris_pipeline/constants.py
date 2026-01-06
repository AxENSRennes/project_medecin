"""Constants and column definitions for BORIS behavioral observation data."""

# Sheet names
TIME_BUDGET_SHEET = "Time budget"
AGGREGATED_SHEET = "Aggregated events"

# Aggregated file suffix patterns (handle inconsistent naming)
AGGREGATED_PATTERNS = ["_agregated", "-agregated", "_aggregated", "-aggregated"]

# Time Budget sheet columns (original files - 16 columns)
TIME_BUDGET_COLUMNS = [
    "Observation id",
    "Observation date",
    "Description",
    "Time budget start (s)",
    "Time budget stop (s)",
    "Time budget duration (s)",
    "Subject",
    "Behavior",
    "Modifiers",
    "Total number of occurences",
    "Total duration (s)",
    "Duration mean (s)",
    "Duration std dev",
    "inter-event intervals mean (s)",
    "inter-event intervals std dev",
    "% of total length",
]

# Aggregated events sheet columns (24 columns)
AGGREGATED_COLUMNS = [
    "Observation id",
    "Observation date",
    "Description",
    "Observation type",
    "Source",
    "Time offset (s)",
    "Coding duration",
    "Media duration (s)",
    "FPS (frame/s)",
    "Subject",
    "Observation duration by subject by observation",
    "Behavior",
    "Behavioral category",
    "Behavior type",
    "Start (s)",
    "Stop (s)",
    "Duration (s)",
    "Media file name",
    "Image index start",
    "Image index stop",
    "Image file path start",
    "Image file path stop",
    "Comment start",
    "Comment stop",
]
