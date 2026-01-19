"""Post-processing functions for BORIS behavioral observation data.

This module provides functions to clean and transform BORIS data after loading,
including column removal, behavior column splitting, and NaN handling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


class BorisPostprocessor:
    """Post-process BORIS data: remove columns, split Behavior, handle NaN."""

    def __init__(
        self,
        file_type: Literal["time_budget", "aggregated"],
    ):
        """Initialize post-processor for a specific BORIS file type.

        Args:
            file_type: Type of BORIS file ('time_budget' or 'aggregated')
        """
        self.file_type = file_type

        # Define columns to remove based on file type
        if file_type == "time_budget":
            self.columns_to_remove = [
                "Observation id",
                "Observation date",
                "Description",
                "Time budget start",
                "Time budget stop",
                "Time budget duration",
                "Subject",
                "Modifiers",
            ]
            # Also check for variations with (s) suffix
            self.columns_to_remove.extend([
                "Time budget start (s)",
                "Time budget stop (s)",
                "Time budget duration (s)",
            ])
            self.behavior_separators = ["-", "_"]  
        else:  # aggregated
            self.columns_to_remove = [
                "Observation id",
                "Observation date",
                "Observation type",
                "Description",
                "Time offset (s)",
                "Coding duration",
                "Media duration (s)",
                "FPS (frame/s)",
                "Source",
                "Subject",
                "Observation duration by subject by observation",
                "Media file name",
                "Image index start",
                "Image index stop",
                "Image file path start",
                "Image file path stop",
                "Comment start",
                "Comment stop",
            ]
            self.behavior_separators = ["-", "_"]  # Dash or underscore for aggregated

    def remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove specified columns from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with columns removed
        """
        df = df.copy()
        # Remove columns that exist
        cols_to_drop = [col for col in self.columns_to_remove if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def split_behavior_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split Behavior column into two columns.

        For time_budget: splits on '-' only
        For aggregated: splits on '-' or '_'

        If no separator found:
        - First column gets "autre"
        - Second column gets the original value

        Args:
            df: Input DataFrame with 'Behavior' column

        Returns:
            DataFrame with 'Behavior_category' and 'Behavior_detail' columns
        """
        df = df.copy()

        if "Behavior" not in df.columns:
            return df

        def split_behavior(value):
            """Split behavior value based on separators."""
            if pd.isna(value):
                return "autre", ""

            value_str = str(value)

            # Try each separator
            for sep in self.behavior_separators:
                if sep in value_str:
                    parts = value_str.split(sep, 1)  # Split only on first occurrence
                    if len(parts) == 2:
                        return parts[0].strip(), parts[1].strip()

            # No separator found
            return "autre", value_str

        # Apply splitting
        behavior_split = df["Behavior"].apply(split_behavior)
        df["Behavior_category"] = [x[0] for x in behavior_split]
        df["Behavior_detail"] = [x[1] for x in behavior_split]

        # Remove original Behavior column
        df = df.drop(columns=["Behavior"])

        return df

    def fill_numeric_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in numeric columns with 0.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with NaN filled in numeric columns
        """
        df = df.copy()

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Fill NaN with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def merge_zero_duration_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Si exactement deux lignes TA. ont une durée nulle, on les fusionne en une seule."""
        if self.file_type == "time_budget":
            return df
        else:
            required_cols = {"Behavior_category", "Start (s)", "Duration (s)"}

            if not required_cols.issubset(df.columns):
                return df  # on ne touche pas si colonnes manquantes

            ta_rows = df[df["Behavior_category"] == "TA."]
            if len(ta_rows) == 2 and (ta_rows["Duration (s)"] == 0).all():
                start = ta_rows["Start (s)"].min()
                if "Stop (s)" in df.columns:
                    stop = ta_rows["Stop (s)"].max()
                else:
                    # À défaut, on suppose stop=start pour chaque ligne, donc on ne peut pas étendre
                    stop = ta_rows["Start (s)"].max()

                duration = stop - start
                new_row = ta_rows.iloc[0].copy()
                new_row["Start (s)"] = start
                if "Stop (s)" in df.columns:
                    new_row["Stop (s)"] = stop
                new_row["Duration (s)"] = duration

                # On supprime les deux anciennes lignes TA. et on ajoute la ligne fusionnée
                df = df[df["Behavior_category"] != "TA."]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            return df
    def enforce_test_start_before_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force Start(Test Gateau en cours) <= Start(TA.) en corrigeant Start si nécessaire.

        Si on modifie Start (s), on met aussi à jour Duration (s) (= Stop - Start) si Stop (s) existe.
        """
        if self.file_type == "time_budget":
            return df
        else:
            required = {"Behavior_category", "Behavior_detail", "Start (s)"}
            if not required.issubset(df.columns):
                return df

            # On ne peut recalculer proprement Duration que si Stop (s) est là
            has_stop = "Stop (s)" in df.columns
            has_duration = "Duration (s)" in df.columns

            ta_rows = df[df["Behavior_detail"] == "lecture INITIALE"]
            test_rows = df[df["Behavior_detail"] == "Test Gateau en cours"]

            if ta_rows.empty or test_rows.empty:
                return df

            # Si plusieurs lignes existent (erreurs de saisie), on prend la première par Start min
            ta_idx = ta_rows["Start (s)"].idxmin()
            test_idx = test_rows["Start (s)"].idxmin()

            ta_start = df.loc[ta_idx, "Start (s)"]
            test_start = df.loc[test_idx, "Start (s)"]

            # Si test démarre après TA, on le recale sur TA
            if pd.notna(ta_start) and pd.notna(test_start) and test_start > ta_start:
                df = df.copy()
                df.loc[test_idx, "Start (s)"] = ta_start

                if has_stop and has_duration:
                    stop = df.loc[test_idx, "Stop (s)"]
                    # Durée recalculée, et clamp à >=0 au cas où
                    new_dur = stop - ta_start if pd.notna(stop) else df.loc[test_idx, "Duration (s)"]
                    df.loc[test_idx, "Duration (s)"] = float(max(0.0, new_dur))

            return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all post-processing steps in order.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        df = self.remove_columns(df)
        df = self.split_behavior_column(df)
        df = self.fill_numeric_nan(df)
        df = self.merge_zero_duration_ta(df)
        df = self.enforce_test_start_before_ta(df)  # <-- ajout
        
        return df


def postprocess_boris(
    df: pd.DataFrame,
    file_type: Literal["time_budget", "aggregated"],
) -> pd.DataFrame:
    """Convenience function to post-process BORIS data.

    Args:
        df: Input DataFrame
        file_type: Type of BORIS file ('time_budget' or 'aggregated')

    Returns:
        Processed DataFrame
    """
    processor = BorisPostprocessor(file_type)
    return processor.process(df)

