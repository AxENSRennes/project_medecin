"""Exemples d'utilisation du module batch_processing.

Ce script montre comment utiliser les différentes fonctions de batch_processing
pour traiter les données Tobii et Boris.
"""

from pathlib import Path

from batch_processing import (
    align_tobii_boris_batch,
    process_boris_batch,
    process_full_dataset,
    process_tobii_batch,
)

# =============================================================================
# Exemple 1: Traiter tous les fichiers Tobii
# =============================================================================


def example_process_all_tobii():
    """Traite tous les fichiers Tobii dans data_G et data_L."""
    results = process_tobii_batch(
        parallel=True,  # Traitement en parallèle
        n_jobs=None,  # Utilise tous les cœurs disponibles
        progress=True,  # Affiche une barre de progression
        file_organization="preserve",  # Conserve la structure de dossiers
    )

    print(f"\n{len(results)} fichiers traités")
    print(f"{sum(1 for r in results if r['status'] == 'success')} réussis")
    print(f"{sum(1 for r in results if r['status'] == 'error')} erreurs")


# =============================================================================
# Exemple 2: Traiter un seul fichier Tobii
# =============================================================================


def example_process_single_tobii():
    """Traite un seul fichier Tobii."""
    filepath = Path("Data/data_G/Tobii/G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv")

    results = process_tobii_batch(
        files=[filepath],
        parallel=False,  # Pas besoin de parallélisme pour un seul fichier
        progress=True,
    )

    if results[0]["status"] == "success":
        print(f"Fichier traité avec succès: {results[0]['output_path']}")
    else:
        print(f"Erreur: {results[0]['error']}")


# =============================================================================
# Exemple 3: Traiter un dossier spécifique
# =============================================================================


def example_process_directory():
    """Traite tous les fichiers dans un dossier spécifique."""
    data_dir = Path("Data/data_G/Tobii")

    results = process_tobii_batch(
        data_dirs=[data_dir],
        parallel=True,
        progress=True,
    )

    print(f"Fichiers traités dans {data_dir}: {len(results)}")


# =============================================================================
# Exemple 4: Traiter les fichiers Boris
# =============================================================================


def example_process_boris():
    """Traite tous les fichiers Boris."""
    results = process_boris_batch(
        parallel=True,
        file_type="aggregated",  # Ou "time_budget" ou "auto"
        progress=True,
    )

    print(f"Fichiers Boris traités: {len(results)}")


# =============================================================================
# Exemple 5: Aligner les données Tobii et Boris
# =============================================================================


def example_align_data():
    """Aligne les données Tobii et Boris traitées."""
    results = align_tobii_boris_batch(
        alignment_method="start",  # Ou "end" ou "center"
        time_before_s=0.0,  # Temps avant l'événement à inclure
        time_after_s=0.0,  # Temps après l'événement à inclure
        n_underscores=6,  # Nombre de underscores pour matcher les noms
        parallel=True,
        progress=True,
    )

    print(f"Paires alignées: {len(results)}")


# =============================================================================
# Exemple 6: Pipeline complet
# =============================================================================


def example_full_pipeline():
    """Exécute le pipeline complet: Tobii, Boris, et alignement."""
    results = process_full_dataset(
        process_tobii=True,  # Traiter les fichiers Tobii
        process_boris=True,  # Traiter les fichiers Boris
        align_data=True,  # Aligner les données
        parallel=True,  # Traitement en parallèle
        file_organization="preserve",  # Conserver la structure
        progress=True,  # Afficher les barres de progression
    )

    print("\nRésumé:")
    print(f"  Tobii: {len(results['tobii'])} fichiers")
    print(f"  Boris: {len(results['boris'])} fichiers")
    print(f"  Alignés: {len(results['aligned'])} paires")


# =============================================================================
# Exemple 7: Traitement avec paramètres personnalisés
# =============================================================================


def example_custom_params():
    """Traite avec des paramètres personnalisés pour Tobii."""
    custom_params = {
        "artifact_detection": {
            "method": "both",
            "z_threshold": 3,  # Seuil plus strict
            "iqr_factor": 1.5,
        },
        "resampling": {
            "enabled": False,
        },
        "missing_data": {
            "max_gap_ms": 5000,  # Gaps plus petits
            "method": "interpolate",
        },
        "smoothing": {
            "method": "savgol",  # Savitzky-Golay au lieu de median
            "window_size": 5,
            "poly_order": 2,
        },
        "baseline_correction": {
            "baseline_method": "first_n_seconds",
            "baseline_duration": 30,
        },
    }

    custom_columns = [
        "Pupil diameter left",
        "Pupil diameter right",
        "Gaze point X",
        "Gaze point Y",
    ]

    results = process_tobii_batch(
        columns_to_process=custom_columns,
        params_dict=custom_params,
        parallel=True,
        progress=True,
    )

    print(f"Traitement avec paramètres personnalisés: {len(results)} fichiers")


# =============================================================================
# Exemple 8: Organisation des fichiers en mode "flat"
# =============================================================================


def example_flat_organization():
    """Traite et organise tous les fichiers dans un seul dossier."""
    results = process_tobii_batch(
        output_root=Path("Data/processed_flat"),
        file_organization="flat",  # Tous les fichiers dans le même dossier
        parallel=True,
        progress=True,
    )

    print(f"Fichiers dans un seul dossier: {len(results)}")


if __name__ == "__main__":
    # Décommenter l'exemple que vous voulez exécuter

    # example_process_all_tobii()
    # example_process_single_tobii()
    # example_process_directory()
    # example_process_boris()
    # example_align_data()
    # example_full_pipeline()
    # example_custom_params()
    # example_flat_organization()

    print("Décommentez l'exemple que vous voulez exécuter dans le script.")

