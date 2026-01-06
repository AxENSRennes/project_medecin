# Fix OOM Issue in run_analysis.py

## Problem Summary

Script gets killed at 72% completion due to out-of-memory (OOM) when processing 78 eye-tracking recordings.

**Root Cause**: Memory accumulation pattern
- Loads all 78 recordings at once: ~3.2 GB
- Keeps recordings list in memory throughout entire pipeline
- At 72%, tries to create BORIS pairs list, pushing total memory > 6.1 GB available
- Linux OOM killer terminates process

**System Constraints**:
- Total RAM: 7.6 GB
- Available: 6.1 GB
- Peak usage: ~7 GB (exceeds limit)

## Memory Flow Analysis

```
Line 327: recordings = load_all_recordings()  → 3.2 GB allocated
Line 335: summary_df = build_summary_df()     → Peak ~6 GB (temp pymovements conversions)
Line 388: create_figure1()                    → Iterates all recordings again
Line 394: load_boris_pairs()                  → ⚠️ OOM KILL (tries to build 2nd list)
```

**Key inefficiencies**:
1. `load_all_recordings()` loads everything upfront (not incremental)
2. Recordings list never released
3. Multiple passes over same data
4. `load_boris_pairs()` creates second list with DataFrame references

## Critical Files

- `scripts/run_analysis.py` (main script)
- `src/tobii_pipeline/analysis/compare.py` (load_all_recordings function)
- `src/tobii_pipeline/analysis/group_viz.py` (heatmap computation)

## Solution: Streaming Architecture

**Impact**: Reduces peak memory from ~7 GB to ~200 MB (97% reduction)

### Overview: Two-Pass Strategy

**Key Insight**: Only heatmaps need multiple recordings simultaneously (to accumulate gaze histograms). They only need 2 columns (Gaze X, Y) - 96% memory savings vs full DataFrames!

**Pass 1**: Stream recordings → compute metrics + extract gaze data (X, Y only) → cache
**Pass 2**: Use cached data for all visualizations (no full DataFrames in memory)

### Expected Memory Profile

| Phase | Current | Streaming | Reduction |
|-------|---------|-----------|-----------|
| Load all | 3.2 GB | - | - |
| Pass 1 (per rec) | - | 180 MB | - |
| Pass 1 end | 3.3 GB | 120 MB | 96% |
| Figure 1 | 3.3 GB | 200 MB | 94% |
| BORIS | 6.5 GB | 150 MB | 98% |
| **Peak** | **~7 GB** | **~200 MB** | **97%** |

## Implementation Plan

### Phase 1: Core Helper Functions (scripts/run_analysis.py)

#### 1.1 Create `stream_recordings()` generator

```python
def stream_recordings(data_dirs, nrows=None, progress=True) -> Iterator[tuple[pd.DataFrame, dict, Path]]:
    """Yield (df, metadata, filepath) one at a time."""
```

- Replaces `load_all_recordings()` which returns a list
- Yields recordings one at a time
- Uses existing loader/cleaner/filter functions
- Includes progress bar and error handling

#### 1.2 Create `extract_gaze_data()` helper

```python
def extract_gaze_data(df, metadata, width=1920, height=1080) -> dict | None:
    """Extract only gaze X/Y columns for heatmaps (96% memory saving)."""
```

- Returns dict with: `{group, participant, recording_id, gaze_x, gaze_y}`
- Filters to valid screen coordinates only
- Returns numpy arrays (~1-5 MB vs 80 MB full DataFrame)

#### 1.3 Create gaze cache functions

```python
def save_gaze_cache(gaze_data_list, cache_path):
    """Save to compressed .npz file (~50 MB total)."""

def load_gaze_cache(cache_path) -> list[dict]:
    """Load from .npz file."""
```

### Phase 2: Refactor Main Pipeline (scripts/run_analysis.py)

#### 2.1 Replace `load_all_recordings()` with streaming Pass 1 (lines 325-335)

**OLD**:
```python
recordings = load_all_recordings(DATA_DIRS, nrows=nrows)
summary_df = build_summary_df(recordings)
```

**NEW**:
```python
summary_rows = []
gaze_data_list = []
sample_filepaths = []

for df, metadata, filepath in stream_recordings(DATA_DIRS, nrows=nrows):
    # Compute metrics for this recording
    summary = compute_recording_summary(df)
    summary_rows.append({...})  # Flatten to row dict

    # Extract gaze data (X, Y only)
    gaze_data = extract_gaze_data(df, metadata)
    gaze_data_list.append(gaze_data)

    # Save first 3 filepaths
    if len(sample_filepaths) < 3:
        sample_filepaths.append((filepath, metadata))

    # CRITICAL: Delete DataFrame immediately
    del df
    gc.collect()

summary_df = pd.DataFrame(summary_rows)
save_gaze_cache(gaze_data_list, OUTPUT_DIR / ".gaze_cache.npz")
```

#### 2.2 Update Figure 1 generation (line 388)

**OLD**: `create_figure1(summary_df, recordings, OUTPUT_DIR)`

**NEW**: `create_figure1_streaming(summary_df, gaze_cache_path, OUTPUT_DIR)`

New function loads gaze cache and passes to visualization helper.

#### 2.3 Update BORIS integration (lines 392-400)

**OLD**:
```python
tobii_boris_pairs = load_boris_pairs(recordings, BORIS_DIRS)
create_figure3(tobii_boris_pairs, OUTPUT_DIR)
```

**NEW**: `create_figure3_streaming(DATA_DIRS, BORIS_DIRS, OUTPUT_DIR, nrows)`

New function:
- Streams through recordings again
- Matches BORIS files
- Computes metrics immediately (gaze/pupil per behavior)
- Saves aggregated metrics (not full DataFrames)
- Creates figure from aggregated metrics

#### 2.4 Update sample visualizations (line 405)

**OLD**: `create_sample_visualizations(recordings, OUTPUT_DIR)`

**NEW**: `create_sample_visualizations_streaming(sample_filepaths, OUTPUT_DIR, nrows)`

New function reloads the 3 saved filepaths one at a time.

### Phase 3: Add Visualization Helpers (src/tobii_pipeline/analysis/group_viz.py)

#### 3.1 Add `compute_group_heatmap_from_cache()`

```python
def compute_group_heatmap_from_cache(gaze_data_list, group, width=1920, height=1080, n_bins=100) -> np.ndarray:
    """Compute heatmap from extracted gaze data (not full DataFrames)."""
```

- Filters to specified group
- Creates 2D histogram per participant
- Normalizes and smooths
- Returns combined heatmap array

#### 3.2 Add `create_group_comparison_figure_streaming()`

```python
def create_group_comparison_figure_streaming(summary_df, gaze_data_list, metrics, figsize):
    """Same layout as original, but uses gaze cache instead of recordings."""
```

- Calls `compute_group_heatmap_from_cache()` for Patient/Control/Difference
- Uses summary_df for violin plots (unchanged)
- Identical output to original function

#### 3.3 Add BORIS visualization helpers

```python
def create_behavioral_figure_from_metrics(metrics_df, figsize):
    """Create Figure 3 from pre-computed participant-level metrics."""

def aggregate_behavioral_metrics(metrics_df):
    """Aggregate to (behavior, group, metric) with 95% CI."""

def plot_behavior_metric_comparison(aggregated_df, metric, label, ax):
    """Grouped bar chart for one metric across behaviors."""
```

### Phase 4: Testing Strategy

1. **Test with 10 recordings**: `--nrows 10000` to verify correctness
2. **Monitor memory**: Add print statements showing memory after each phase
3. **Full run**: Remove nrows limit, monitor for completion
4. **Verify outputs**: Compare all CSVs and PNGs to baseline (if available)

### Files to Modify

1. **`scripts/run_analysis.py`** (primary changes)
   - Lines 320-420: Refactor `run_analysis()` function
   - Add: Helper functions (~300 new lines)
   - Add: New figure creation wrappers

2. **`src/tobii_pipeline/analysis/group_viz.py`** (backward-compatible additions)
   - Add: `compute_group_heatmap_from_cache()` (~50 lines)
   - Add: `create_group_comparison_figure_streaming()` (~80 lines)
   - Add: BORIS visualization helpers (~150 lines)
   - Keep original functions unchanged

3. **No changes needed**:
   - `compare.py`: `load_all_recordings()` kept for backward compatibility
   - `integration/cross_modal.py`: Already works with single pairs

## Alternative Quick Wins (if not doing full refactor)

1. **Skip BORIS integration**: `python scripts/run_analysis.py --no-boris`
   - Will likely complete successfully

2. **Process in two runs**:
   - Run 1: Process data_G only (39 files)
   - Run 2: Process data_L only (39 files)
   - Manually combine results

3. **Increase WSL2 memory limit**:
   - Edit `.wslconfig` to allocate more RAM
   - Temporary fix, doesn't solve root cause
