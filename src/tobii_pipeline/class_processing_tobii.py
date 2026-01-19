import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt
from typing import List

# =============================================================================
# Helper function for verbose printing
# =============================================================================

def _vprint(message: str, verbose_level: int, required_level: int):
    """Print message only if verbose_level >= required_level.
    
    Args:
        message: Message to print
        verbose_level: Current verbose level (0, 1, or 2+)
        required_level: Minimum level required to print (1 for steps, 2 for details)
    """
    if verbose_level >= required_level:
        print(message)

class FirstPreprocessing:
    """
    Classe pour le preprocessing préliminaire des données Tobii
    
    Cette classe effectue les étapes de nettoyage initiales :
    1. Suppression de la première ligne et de colonnes non pertinentes
    2. Réduction à 1 ligne sur 3 (agrégation)
    3. Filtrage par validité des yeux
    """
    
    def __init__(self, tobii_data: pd.DataFrame, verbose: int = 2):
        """
        Initialise le preprocesseur
        
        Parameters:
        -----------
        tobii_data : DataFrame
            Données Tobii brutes
        verbose : int
            Niveau de verbosité (0: aucun print, 1: étapes principales, 2+: tous les prints)
        """
        self.data = tobii_data.copy()
        self.timestamp_col = 'Recording timestamp'
        self.verbose = verbose
    
    def step1_remove_first_row_and_columns(self, 
                                          columns_to_remove: List[int] = None,
                                          keep_sensor: bool = True):
        """
        ÉTAPE 1 : Supprime la première ligne et les colonnes spécifiées
        
        Parameters:
        -----------
        columns_to_remove : list of int, optional
            Liste des indices de colonnes à supprimer.
            Par défaut : range(1,15) + range(36,39)
        keep_sensor : bool
            Si True, garde la colonne Sensor même si elle est dans columns_to_remove
            (nécessaire pour step3_reduce_to_one_in_three_with_sensor)
        
        Returns:
        --------
        DataFrame : Données nettoyées
        """
        if columns_to_remove is None:
            # Colonnes par défaut : range(1,15) + range(36,39)
            columns_to_remove = list(range(1, 15)) + list(range(36, 39))
        
        _vprint(f"ÉTAPE 1 - Suppression première ligne et colonnes", self.verbose, 1)
        _vprint(f"  - Lignes avant : {len(self.data)}", self.verbose, 2)
        _vprint(f"  - Colonnes avant : {len(self.data.columns)}", self.verbose, 2)
        _vprint(f"  - Colonnes à supprimer : {len(columns_to_remove)}", self.verbose, 2)
        
        # Supprimer la première ligne
        self.data = self.data.iloc[1:].reset_index(drop=True)
        
        # Supprimer les colonnes spécifiées
        columns_to_drop = [self.data.columns[i] for i in columns_to_remove if i < len(self.data.columns)]
        
        # Ne pas supprimer Sensor si demandé
        if keep_sensor and 'Sensor' in columns_to_drop:
            columns_to_drop.remove('Sensor')
            _vprint(f"  - Colonne Sensor conservée (nécessaire pour step3)", self.verbose, 2)
        
        self.data = self.data.drop(columns=columns_to_drop)
        
        _vprint(f"  - Lignes après : {len(self.data)}", self.verbose, 2)
        _vprint(f"  - Colonnes après : {len(self.data.columns)}", self.verbose, 2)
        _vprint(f"  - Colonnes supprimées : {columns_to_drop}", self.verbose, 2)
        
        return self.data
    
    def step1_5_convert_dtypes(self, 
                            float_precision='float32',
                            timestamp_col='Recording timestamp',
                            int_cols=None,
                            categorical_cols=None):
        """
        ÉTAPE 1.5 : Conversion optimale des types de données
        
        Convertit les colonnes numériques en float32 (optimisé mémoire),
        les colonnes entières en int, et les catégorielles en category.
        
        Position : Après suppression colonnes (step1) mais avant réduction lignes (step2)
        pour optimiser la performance.
        
        Parameters:
        -----------
        float_precision : str
            'float32' (4 bytes, ~7 décimales) ou 'float64' (8 bytes, ~15 décimales)
            Recommandé : float32 pour eye tracking (suffisant et 2x moins de mémoire)
        timestamp_col : str
            Nom de la colonne timestamp
        int_cols : list, optional
            Liste des colonnes à convertir en int (défaut : auto-détecté)
        categorical_cols : list, optional
            Liste des colonnes catégorielles (défaut : auto-détecté)
        
        Returns:
        --------
        DataFrame : Données avec types optimisés
        """
        _vprint(f"\nÉTAPE 1.5 - Conversion des types de données", self.verbose, 1)
        _vprint(f"  - Précision float : {float_precision}", self.verbose, 2)
        
        # Colonnes à convertir en int (par défaut)
        if int_cols is None:
            int_cols = [timestamp_col, 'Eye movement type index']
            # Filtrer celles qui existent
            int_cols = [col for col in int_cols if col in self.data.columns]
        
        # Colonnes catégorielles (par défaut)
        if categorical_cols is None:
            categorical_cols = ['Validity left', 'Validity right', 'Eye movement type']
            # Filtrer celles qui existent
            categorical_cols = [col for col in categorical_cols if col in self.data.columns]
        
        # Mémoire avant
        memory_before = self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Convertir colonnes entières
        for col in int_cols:
            if col in self.data.columns:
                try:
                    # Convertir en numérique puis arrondir (pour gérer les .0)
                    numeric = pd.to_numeric(self.data[col], errors='coerce')
                    # Arrondir pour éliminer les .0
                    numeric = numeric.round(0)
                    self.data[col] = numeric.astype('Int64')  # Int64 supporte les NaN
                    _vprint(f"  - {col} → Int64", self.verbose, 2)
                except Exception as e:
                    _vprint(f"  ⚠️ Impossible de convertir {col} en int : {e}", self.verbose, 2)
        
        # Convertir colonnes catégorielles
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('category')
                _vprint(f"  - {col} → category", self.verbose, 2)
        
        # Convertir toutes les autres colonnes numériques en float
        numeric_cols = self.data.select_dtypes(include=[np.number, 'object']).columns
        numeric_cols = [col for col in numeric_cols 
                       if col not in int_cols and col not in categorical_cols 
                       and col != timestamp_col]
        
        converted_count = 0
        for col in numeric_cols:
            try:
                # Essayer de convertir en numérique (gère les virgules)
                converted = pd.to_numeric(
                    self.data[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
                
                # Vérifier si conversion réussie (au moins quelques valeurs)
                if not converted.isna().all():
                    self.data[col] = converted.astype(float_precision)
                    converted_count += 1
            except:
                pass
        
        _vprint(f"  - Colonnes numériques converties : {converted_count}", self.verbose, 2)
        
        # Mémoire après
        memory_after = self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_saved = memory_before - memory_after
        
        _vprint(f"  - Mémoire avant : {memory_before:.2f} MB", self.verbose, 2)
        _vprint(f"  - Mémoire après : {memory_after:.2f} MB", self.verbose, 2)
        _vprint(f"  - Mémoire économisée : {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}%)", self.verbose, 2)
        
        return self.data
    
    def step2_reduce_to_one_in_three(self):
        """
        ÉTAPE 2 (FALLBACK) : Réduit à 1 ligne sur 3 avec méthode modulo 3
        
        Méthode simple basée sur la position (modulo 3) - utilisé si Sensor n'est pas disponible
        
        Returns:
        --------
        DataFrame : Données réduites
        """
        _vprint(f"\nÉTAPE 2 (FALLBACK) - Réduction modulo 3", self.verbose, 1)
        _vprint(f"  - Lignes avant : {len(self.data)}", self.verbose, 2)
        
        # Grouper par blocs de 3
        n_groups = len(self.data) // 3
        remainder = len(self.data) % 3
        
        reduced_rows = []
        
        for i in range(n_groups):
            start_idx = i * 3
            end_idx = start_idx + 3
            group = self.data.iloc[start_idx:end_idx]
            
            # Créer nouvelle ligne
            new_row = {}
            
            # Pour chaque colonne
            for col in self.data.columns:
                col_values = group[col].values
                
                # Timestamp : toujours garder celui de la première ligne
                if col == self.timestamp_col:
                    new_row[col] = col_values[0]
                    continue
                
                # Enlever les NaN pour l'analyse
                non_nan_values = col_values[~pd.isna(col_values)]
                
                # Si que des NaN
                if len(non_nan_values) == 0:
                    new_row[col] = np.nan
                    continue
                
                # Vérifier si c'est numérique ou catégoriel
                is_numeric = False
                try:
                    # Essayer de convertir en numérique
                    numeric_values = pd.to_numeric(
                        non_nan_values.astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
                    if not numeric_values.isna().all():
                        is_numeric = True
                        non_nan_values = numeric_values.dropna()
                except:
                    pass
                
                if is_numeric:
                    # Valeurs numériques
                    unique_values = non_nan_values.unique()
                    if len(unique_values) == 1:
                        # Valeur constante
                        new_row[col] = unique_values[0]
                    else:
                        # Valeurs différentes : moyenne
                        new_row[col] = non_nan_values.mean()
                else:
                    # Valeurs catégorielles
                    unique_values = non_nan_values
                    if len(unique_values) == 1:
                        # Valeur constante
                        new_row[col] = unique_values[0]
                    else:
                        # Valeur majoritaire
                        from collections import Counter
                        counter = Counter(unique_values.astype(str))
                        new_row[col] = counter.most_common(1)[0][0]
            
            reduced_rows.append(new_row)
        
        # Gérer le reste (lignes qui ne font pas un groupe de 3)
        if remainder > 0:
            remainder_group = self.data.iloc[n_groups * 3:]
            # Pour le reste, on peut soit les garder telles quelles, soit les ignorer
            # Ici on les garde telles quelles (première ligne du reste)
            if len(remainder_group) > 0:
                remainder_row = remainder_group.iloc[0].to_dict()
                reduced_rows.append(remainder_row)
        
        # Créer nouveau DataFrame
        self.data = pd.DataFrame(reduced_rows)
        
        _vprint(f"  - Lignes après : {len(self.data)}", self.verbose, 2)
        if n_groups > 0:
            _vprint(f"  - Réduction : {len(self.data) / (n_groups * 3 + remainder) * 100:.1f}%", self.verbose, 2)
        
        return self.data

    def step3_reduce_to_one_in_three_with_sensor(self, sensor_col='Sensor'):
        """
        ÉTAPE 2 (AMÉLIORÉE) : Réduit à 1 ligne sur 3 en utilisant la colonne Sensor
        
        Groupe les données par capteur (Eye tracker, Gyroscope, Accelerometer)
        et agrège sur le timestamp de l'Eye tracker.
        
        Parameters:
        -----------
        sensor_col : str
            Nom de la colonne Sensor
        
        Returns:
        --------
        DataFrame : Données réduites
        """
        _vprint(f"\nÉTAPE 2 - Réduction avec colonne Sensor", self.verbose, 1)
        _vprint(f"  - Lignes avant : {len(self.data)}", self.verbose, 2)
        _vprint(f"  - Colonnes disponibles : {list(self.data.columns)}", self.verbose, 2)
        
        if sensor_col not in self.data.columns:
            _vprint(f"  ⚠️ Colonne {sensor_col} non trouvée dans les colonnes disponibles", self.verbose, 2)
            _vprint(f"  ⚠️ Utilisation méthode modulo 3 (fallback)", self.verbose, 2)
            return self.step2_reduce_to_one_in_three()  # Fallback
        
        # Vérifier les valeurs uniques de Sensor
        sensor_values = self.data[sensor_col].value_counts()
        _vprint(f"  - Valeurs Sensor détectées : {sensor_values.to_dict()}", self.verbose, 2)
        _vprint(f"  - Nombre de valeurs uniques : {len(sensor_values)}", self.verbose, 2)
        
        # Identifier les 3 types de capteurs depuis value_counts
        sensor_types = sensor_values.index.tolist()
        if len(sensor_types) != 3:
            _vprint(f"  ⚠️ ERREUR : Attendu 3 types de capteurs, trouvé {len(sensor_types)}", self.verbose, 2)
            _vprint(f"  ⚠️ Types trouvés : {sensor_types}", self.verbose, 2)
            raise ValueError(f"Nombre incorrect de types de capteurs : {len(sensor_types)} au lieu de 3")
        
        # Identifier quel capteur est lequel
        eye_tracker_type = None
        gyro_type = None
        acc_type = None
        
        for sensor_type in sensor_types:
            sensor_lower = str(sensor_type).lower()
            if 'eye' in sensor_lower or 'tracker' in sensor_lower or 'gaze' in sensor_lower:
                eye_tracker_type = sensor_type
            elif 'gyro' in sensor_lower:
                gyro_type = sensor_type
            elif 'accel' in sensor_lower:
                acc_type = sensor_type
        
        # Vérifier qu'on a trouvé les 3
        if not all([eye_tracker_type, gyro_type, acc_type]):
            _vprint(f"  ⚠️ Impossible d'identifier les 3 capteurs automatiquement", self.verbose, 2)
            _vprint(f"  - Eye tracker : {eye_tracker_type}", self.verbose, 2)
            _vprint(f"  - Gyroscope : {gyro_type}", self.verbose, 2)
            _vprint(f"  - Accelerometer : {acc_type}", self.verbose, 2)
            # Utiliser l'ordre des value_counts comme fallback
            eye_tracker_type = sensor_types[0]
            gyro_type = sensor_types[1]
            acc_type = sensor_types[2]
            _vprint(f"  - Utilisation ordre value_counts : {eye_tracker_type}, {gyro_type}, {acc_type}", self.verbose, 2)
        
        _vprint(f"  - Eye tracker identifié : {eye_tracker_type}", self.verbose, 2)
        _vprint(f"  - Gyroscope identifié : {gyro_type}", self.verbose, 2)
        _vprint(f"  - Accelerometer identifié : {acc_type}", self.verbose, 2)
        
        # Normaliser les noms de capteurs pour détection
        def normalize_sensor_name(sensor_str):
            """Normalise le nom du capteur pour détection"""
            sensor_str = str(sensor_str).strip()
            if sensor_str == eye_tracker_type:
                return 'eye_tracker'
            elif sensor_str == gyro_type:
                return 'gyroscope'
            elif sensor_str == acc_type:
                return 'accelerometer'
            else:
                # Fallback sur détection par mots-clés
                sensor_lower = sensor_str.lower()
                if 'eye' in sensor_lower or 'tracker' in sensor_lower or 'gaze' in sensor_lower:
                    return 'eye_tracker'
                elif 'gyro' in sensor_lower:
                    return 'gyroscope'
                elif 'accel' in sensor_lower:
                    return 'accelerometer'
                return sensor_lower
        
        # Identifier les groupes de 3 lignes avec les 3 capteurs
        self.data = self.data.sort_values(by=self.timestamp_col).reset_index(drop=True)
        
        reduced_rows = []
        i = 0
        groups_found = 0
        skipped_lines = 0
        max_iterations = len(self.data)  # Protection contre boucle infinie
        iterations = 0
        
        while i < len(self.data) and iterations < max_iterations:
            iterations += 1
            
            # Chercher un groupe de 3 lignes avec les 3 capteurs EXACTS
            group = {}
            # Utiliser les valeurs exactes de Sensor
            group[eye_tracker_type] = None
            group[gyro_type] = None
            group[acc_type] = None
            
            # Chercher les 3 capteurs dans une fenêtre temporelle
            window_size = 10  # Fenêtre de recherche
            for j in range(i, min(i + window_size, len(self.data))):
                sensor_val = str(self.data.iloc[j][sensor_col]).strip()
                
                # Vérifier si c'est un des 3 capteurs qu'on cherche
                if sensor_val == eye_tracker_type and group[eye_tracker_type] is None:
                    group[eye_tracker_type] = j
                elif sensor_val == gyro_type and group[gyro_type] is None:
                    group[gyro_type] = j
                elif sensor_val == acc_type and group[acc_type] is None:
                    group[acc_type] = j
                
                # Si on a trouvé les 3, arrêter
                if all(v is not None for v in group.values()):
                    break
            
            # Vérifier si on a trouvé les 3 capteurs
            if all(v is not None for v in group.values()):
                # Créer liste d'indices dans l'ordre Eye tracker, Gyro, Acc
                group_indices = [group[eye_tracker_type], group[gyro_type], group[acc_type]]
                group_data = self.data.iloc[group_indices]
                
                # Créer nouvelle ligne
                new_row = {}
                
                # Timestamp : utiliser celui de l'Eye tracker (premier du groupe)
                eye_tracker_idx = group[eye_tracker_type]
                base_timestamp = self.data.iloc[eye_tracker_idx][self.timestamp_col]
                
                # Pour chaque colonne
                for col in self.data.columns:
                    if col == self.timestamp_col:
                        new_row[col] = base_timestamp
                        continue
                    if col == sensor_col:
                        # Ne pas inclure Sensor dans la ligne agrégée
                        continue
                    
                    # Récupérer les valeurs du groupe
                    col_values = group_data[col].values
                    
                    # Enlever les NaN
                    non_nan_values = col_values[~pd.isna(col_values)]
                    
                    if len(non_nan_values) == 0:
                        new_row[col] = np.nan
                        continue
                    
                    # Vérifier si numérique ou catégoriel
                    is_numeric = False
                    try:
                        numeric_vals = pd.to_numeric(
                            non_nan_values.astype(str).str.replace(',', '.'), 
                            errors='coerce'
                        )
                        if not numeric_vals.isna().all():
                            is_numeric = True
                            non_nan_values = numeric_vals.dropna()
                    except:
                        pass
                    
                    if is_numeric:
                        # Numérique : moyenne si différentes, valeur si constante
                        unique_vals = non_nan_values.unique()
                        if len(unique_vals) == 1:
                            new_row[col] = unique_vals[0]
                        else:
                            new_row[col] = non_nan_values.mean()
                    else:
                        # Catégoriel : valeur majoritaire
                        from collections import Counter
                        counter = Counter(non_nan_values.astype(str))
                        new_row[col] = counter.most_common(1)[0][0]
                
                reduced_rows.append(new_row)
                groups_found += 1
                # Avancer après le dernier index du groupe
                i = max(group_indices) + 1
            else:
                # Pas de groupe complet trouvé, avancer d'une ligne
                skipped_lines += 1
                i += 1
        
        # Vérifier si des groupes ont été trouvés
        if len(reduced_rows) == 0:
            _vprint(f"\n  ⚠️ ERREUR : Aucun groupe de 3 capteurs trouvé !", self.verbose, 2)
            _vprint(f"  - Lignes parcourues : {i}", self.verbose, 2)
            _vprint(f"  - Lignes sautées : {skipped_lines}", self.verbose, 2)
            _vprint(f"  - Valeurs Sensor uniques : {sensor_values.index.tolist()}", self.verbose, 2)
            _vprint(f"  - Utilisation méthode modulo 3 (fallback)", self.verbose, 2)
            return self.step2_reduce_to_one_in_three()  # Fallback
        
        # Créer nouveau DataFrame
        self.data = pd.DataFrame(reduced_rows)
        
        _vprint(f"  - Groupes de 3 capteurs trouvés : {groups_found}", self.verbose, 2)
        _vprint(f"  - Lignes après : {len(self.data)}", self.verbose, 2)
        if len(reduced_rows) > 0:
            reduction_pct = len(self.data) / (len(self.data) * 3) * 100
            _vprint(f"  - Réduction : ~{reduction_pct:.1f}% (1 ligne sur 3)", self.verbose, 2)
        
        return self.data

    def step2_filter_invalid_eyes(self, 
                                  validity_left_col: str = 'Validity left',
                                  validity_right_col: str = 'Validity right',
                                  require_both_valid: bool = False,
                                  sensor_col: str = 'Sensor'):
        """
        ÉTAPE 3 : Filtre les lignes selon la validité des yeux
        
        IMPORTANT : Si l'Eye tracker est invalide, supprime aussi les lignes
        Gyroscope et Accelerometer du même groupe (identifiées via colonne Sensor).
        
        Parameters:
        -----------
        validity_left_col : str
            Nom de la colonne de validité œil gauche
        validity_right_col : str
            Nom de la colonne de validité œil droit
        require_both_valid : bool
            Si True : garde seulement si les DEUX yeux valides
            Si False : garde si AU MOINS UN œil valide (défaut)
        sensor_col : str
            Nom de la colonne Sensor pour identifier les groupes de 3
        
        Returns:
        --------
        DataFrame : Données filtrées
        """
        _vprint(f"\nÉTAPE 3 - Filtrage par validité des yeux", self.verbose, 1)
        _vprint(f"  - Lignes avant : {len(self.data)}", self.verbose, 2)
        
        if validity_left_col not in self.data.columns or \
           validity_right_col not in self.data.columns:
            _vprint(f"  ⚠️ Colonnes de validité non trouvées - pas de filtrage", self.verbose, 2)
            return self.data
        
        # Convertir validité en booléen
        def is_valid(val):
            """Convertit une valeur de validité en booléen"""
            if pd.isna(val):
                return False
            val_str = str(val).lower()
            if 'invalid' in val_str:
                return False
            if 'valid' in val_str:
                return True
            # Essayer conversion numérique (0 = valide, >0 = invalide)
            try:
                num_val = float(val)
                return num_val == 0
            except:
                return False
        
        # Convertir en booléen directement (nécessaire pour éviter TypeError avec Categorical)
        validity_left = self.data[validity_left_col].apply(is_valid).astype(bool)
        validity_right = self.data[validity_right_col].apply(is_valid).astype(bool)
        
        # Identifier les lignes Eye tracker invalides
        if require_both_valid:
            eye_tracker_invalid = ~(validity_left & validity_right)
            criterion = "les DEUX yeux valides"
        else:
            eye_tracker_invalid = ~(validity_left | validity_right)
            criterion = "AU MOINS UN œil valide"
        
        # Si Sensor est disponible, supprimer les groupes complets (Eye tracker + Gyro + Acc)
        if sensor_col in self.data.columns:
            _vprint(f"  - Utilisation colonne {sensor_col} pour supprimer groupes complets", self.verbose, 2)
            
            # Trier les données par timestamp pour avoir l'ordre
            self.data = self.data.sort_values(by=self.timestamp_col).reset_index(drop=True)
            
            # Recréer le masque après tri et convertir en booléen (nécessaire pour éviter TypeError avec Categorical)
            validity_left = self.data[validity_left_col].apply(is_valid).astype(bool)
            validity_right = self.data[validity_right_col].apply(is_valid).astype(bool)
            
            if require_both_valid:
                eye_tracker_invalid = ~(validity_left & validity_right)
            else:
                eye_tracker_invalid = ~(validity_left | validity_right)
            
            # Identifier les 3 types de capteurs depuis Sensor
            sensor_values = self.data[sensor_col].value_counts()
            sensor_types = sensor_values.index.tolist()
            
            if len(sensor_types) != 3:
                _vprint(f"  ⚠️ Nombre de types Sensor incorrect : {len(sensor_types)}", self.verbose, 2)
                _vprint(f"  ⚠️ Utilisation filtrage simple", self.verbose, 2)
                valid_mask = ~eye_tracker_invalid
            else:
                # Identifier quel capteur est lequel
                eye_tracker_type = None
                gyro_type = None
                acc_type = None
                
                for sensor_type in sensor_types:
                    sensor_lower = str(sensor_type).lower()
                    if 'eye' in sensor_lower or 'tracker' in sensor_lower or 'gaze' in sensor_lower:
                        eye_tracker_type = sensor_type
                    elif 'gyro' in sensor_lower:
                        gyro_type = sensor_type
                    elif 'accel' in sensor_lower:
                        acc_type = sensor_type
                
                # Fallback : utiliser l'ordre si identification échoue
                if not all([eye_tracker_type, gyro_type, acc_type]):
                    eye_tracker_type = sensor_types[0]
                    gyro_type = sensor_types[1]
                    acc_type = sensor_types[2]
                
                _vprint(f"  - Types Sensor : Eye tracker={eye_tracker_type}, Gyro={gyro_type}, Acc={acc_type}", self.verbose, 2)
                
                # Identifier les indices des lignes Eye tracker invalides
                invalid_eye_tracker_indices = []
                for idx in range(len(self.data)):
                    if eye_tracker_invalid.iloc[idx] and str(self.data.iloc[idx][sensor_col]).strip() == str(eye_tracker_type).strip():
                        invalid_eye_tracker_indices.append(idx)
                
                # Pour chaque Eye tracker invalide, trouver et marquer les 2 lignes suivantes (Gyro + Acc)
                indices_to_remove = set(invalid_eye_tracker_indices)
                
                for eye_idx in invalid_eye_tracker_indices:
                    # Chercher les 2 lignes suivantes avec Gyro et Acc dans une fenêtre
                    window_size = 5
                    start_search = eye_idx + 1
                    end_search = min(eye_idx + window_size, len(self.data))
                    
                    gyro_found = False
                    acc_found = False
                    
                    for j in range(start_search, end_search):
                        if j >= len(self.data):
                            break
                        sensor_val = str(self.data.iloc[j][sensor_col]).strip()
                        
                        if sensor_val == str(gyro_type).strip() and not gyro_found:
                            indices_to_remove.add(j)
                            gyro_found = True
                        elif sensor_val == str(acc_type).strip() and not acc_found:
                            indices_to_remove.add(j)
                            acc_found = True
                        
                        if gyro_found and acc_found:
                            break  # On a trouvé Gyro et Acc
                
                _vprint(f"  - Lignes Eye tracker invalides : {len(invalid_eye_tracker_indices)}", self.verbose, 2)
                _vprint(f"  - Groupes complets à supprimer : {len(invalid_eye_tracker_indices)}", self.verbose, 2)
                _vprint(f"  - Total lignes à supprimer : {len(indices_to_remove)}", self.verbose, 2)
                
                # Créer masque final
                valid_mask = ~self.data.index.isin(indices_to_remove)
            
            _vprint(f"  - Lignes Eye tracker invalides : {len(invalid_eye_tracker_indices)}", self.verbose, 2)
            _vprint(f"  - Groupes complets à supprimer : {len(invalid_eye_tracker_indices)}", self.verbose, 2)
            _vprint(f"  - Total lignes à supprimer : {len(indices_to_remove)}", self.verbose, 2)
            
        else:
            # Pas de Sensor : filtrage simple (ancienne méthode)
            _vprint(f"  ⚠️ Colonne {sensor_col} non trouvée - filtrage simple", self.verbose, 2)
            valid_mask = ~eye_tracker_invalid
        
        n_before = len(self.data)
        self.data = self.data[valid_mask].reset_index(drop=True)
        n_after = len(self.data)
        
        n_removed = n_before - n_after
        removal_pct = (n_removed / n_before) * 100 if n_before > 0 else 0
        
        _vprint(f"  - Critère : {criterion}", self.verbose, 2)
        _vprint(f"  - Lignes après : {len(self.data)}", self.verbose, 2)
        _vprint(f"  - Lignes supprimées : {n_removed} ({removal_pct:.2f}%)", self.verbose, 2)
        
        if removal_pct > 50:
            _vprint(f"  ⚠️ ATTENTION : Plus de 50% des données supprimées", self.verbose, 2)
        
        return self.data
    
    def step4_replace_fixation_nan_with_zero(self,
                                        fixation_x_col: str = 'Fixation point X',
                                        fixation_y_col: str = 'Fixation point Y'):
        """
        ÉTAPE 4 : Remplace les NaN de Fixation point X et Y par 0
        
        Convention : 0 signifie "ne fixe pas" (pas de fixation détectée)
        
        Parameters:
        -----------
        fixation_x_col : str
            Nom de la colonne Fixation point X
        fixation_y_col : str
            Nom de la colonne Fixation point Y
        
        Returns:
        --------
        DataFrame : Données avec NaN remplacés par 0
        """
        _vprint(f"\nÉTAPE 4 - Remplacement NaN Fixation point par 0", self.verbose, 1)
        
        nan_replaced_x = 0
        nan_replaced_y = 0
        
        if fixation_x_col in self.data.columns:
            nan_count_x = self.data[fixation_x_col].isna().sum()
            self.data[fixation_x_col] = self.data[fixation_x_col].fillna(0)
            nan_replaced_x = nan_count_x
            _vprint(f"  - {fixation_x_col}: {nan_replaced_x} NaN remplacés par 0", self.verbose, 2)
        else:
            _vprint(f"  ⚠️ Colonne {fixation_x_col} non trouvée", self.verbose, 2)
        
        if fixation_y_col in self.data.columns:
            nan_count_y = self.data[fixation_y_col].isna().sum()
            self.data[fixation_y_col] = self.data[fixation_y_col].fillna(0)
            nan_replaced_y = nan_count_y
            _vprint(f"  - {fixation_y_col}: {nan_replaced_y} NaN remplacés par 0", self.verbose, 2)
        else:
            _vprint(f"  ⚠️ Colonne {fixation_y_col} non trouvée", self.verbose, 2)
        
        total_replaced = nan_replaced_x + nan_replaced_y
        if total_replaced > 0:
            _vprint(f"  - Total NaN remplacés : {total_replaced}", self.verbose, 2)
            _vprint(f"  - Convention : 0 = pas de fixation détectée", self.verbose, 2)
        
        return self.data

    def apply_all_steps(self, 
                   columns_to_remove: List[int] = None,
                   require_both_valid: bool = False,
                   float_precision: str = 'float32'):
        """
        Applique toutes les étapes de preprocessing préliminaire
        
        Parameters:
        -----------
        columns_to_remove : list of int, optional
            Colonnes à supprimer (défaut : range(1,15) + range(36,39))
        require_both_valid : bool
            Si True, garde seulement si les deux yeux valides
        convert_dtypes : bool
            Si True, convertit les types de données (défaut: True)
        float_precision : str
            'float32' ou 'float64' (défaut: 'float32')
        """
        _vprint(f"{'='*80}", self.verbose, 2)
        _vprint(f"PREPROCESSING PRÉLIMINAIRE COMPLET", self.verbose, 2)
        _vprint(f"{'='*80}", self.verbose, 2)
        
        # Étape 1 : Suppression colonnes (garder Sensor pour step3)
        self.step1_remove_first_row_and_columns(columns_to_remove, keep_sensor=True)
        
        # Étape 1.5 : Conversion types (OPTIMAL : après step1, avant step2)
        self.step1_5_convert_dtypes(float_precision=float_precision)
        
        # Étape 2 : Filtrage validité
        self.step2_filter_invalid_eyes(require_both_valid=require_both_valid)
        
        # Étape 3 : Réduction 1/3
        self.step3_reduce_to_one_in_three_with_sensor()
        
        
        # Étape 4 : Fixation point NaN → 0
        self.step4_replace_fixation_nan_with_zero()
        
        _vprint(f"\n{'='*80}", self.verbose, 2)
        _vprint(f"PREPROCESSING PRÉLIMINAIRE TERMINÉ", self.verbose, 2)
        _vprint(f"{'='*80}", self.verbose, 2)
        _vprint(f"  - Lignes finales : {len(self.data)}", self.verbose, 2)
        _vprint(f"  - Colonnes finales : {len(self.data.columns)}", self.verbose, 2)
        
        return self.data


class TobiiTimeSeriesPreprocessor:
    """
    Pipeline de preprocessing pour séries temporelles Tobii
    
    Ce pipeline implémente les bonnes pratiques de preprocessing pour les données
    d'eye tracking Tobii, en particulier pour la mesure de la fatigabilité cognitive.
    Il gère l'échantillonnage IRRÉGULIER (événementiel) caractéristique des données Tobii.
    
    Références principales:
    ----------
    Kret, M. E., & Sjak-Shie, E. E. (2018). Preprocessing pupil size data: 
    Guidelines and code. Behavior Research Methods, 51(4), 1336-1342.
    https://doi.org/10.3758/s13428-018-1075-y
    
    Méthodes utilisées:
    ----------
    - Détection d'artefacts : Z-score (seuil=3) + IQR (facteur=1.5)
      Justification : Méthode standard robuste pour détecter les valeurs aberrantes
      
    - Gestion des données manquantes : Interpolation linéaire basée sur timestamps réels
      Justification : Préserve les relations temporelles dans l'échantillonnage irrégulier
      
    - Rééchantillonnage : Auto-détection de l'intervalle optimal (médiane des intervalles normaux)
      Justification : Nécessaire pour lissage Savitzky-Golay et synchronisation multi-capteurs
      
    - Lissage : Savitzky-Golay (fenêtre adaptative selon signal) ou filtre médian
      Justification : Savitzky-Golay préserve mieux les caractéristiques du signal que la 
      moyenne mobile (Kret & Sjak-Shie, 2018). Filtre médian pour signaux avec pics (IMU).
      
    - Correction baseline : Premières N secondes pour pupille, moyenne pour IMU
      Justification : Standard en pupillométrie pour normaliser les variations individuelles
    """
    
    def __init__(self, tobii_data, timestamp_col=None, verbose: int = 2):
        """
        Initialise le preprocesseur
        
        Parameters:
        -----------
        tobii_data : DataFrame
            Données Tobii avec colonnes temporelles et mesures
        timestamp_col : str, optional
            Nom de la colonne timestamp (par défaut: première colonne)
        verbose : int
            Niveau de verbosité (0: aucun print, 1: étapes principales, 2+: tous les prints)
        """
        self.data = tobii_data.copy()
        self.timestamp_col = timestamp_col if timestamp_col else tobii_data.columns[0]
        self.processed_data = None
        self.preprocessing_report = {}
        self.uniform_timestamps = None  # Index temporel uniforme (créé une fois)
        self.target_interval_ms = None  # Intervalle cible (déterminé une fois)
        self.verbose = verbose
        
        # Vérifier si l'échantillonnage est irrégulier
        self._check_sampling_regularity()
    
    def _check_sampling_regularity(self):
        """Vérifie si l'échantillonnage est régulier ou irrégulier"""
        timestamps = self.data[self.timestamp_col].values
        intervals = np.diff(timestamps)
        
        # Coefficient de variation des intervalles
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        self.is_regular = cv < 0.1  # Si CV < 10%, considéré comme régulier
        self.median_interval = np.median(intervals)
        
        _vprint(f"Analyse de l'échantillonnage:", self.verbose, 1)
        _vprint(f"  - Échantillonnage régulier : {self.is_regular}", self.verbose, 2)
        _vprint(f"  - Intervalle médian : {self.median_interval:.2f} ms", self.verbose, 2)
        _vprint(f"  - Coefficient de variation : {cv:.2%}", self.verbose, 2)
        
        if not self.is_regular:
            _vprint(f"  ⚠️ Échantillonnage IRRÉGULIER détecté - rééchantillonnage recommandé", self.verbose, 2)
    
    def step1_detect_artifacts(self, column, method='statistical', 
                               z_threshold=3, iqr_factor=1.5):
        """
        ÉTAPE 1 : Détection des artefacts
        (Fonctionne avec échantillonnage irrégulier - pas de changement)
        """
        if isinstance(column, int):
            col_name = self.data.columns[column]
        else:
            col_name = column
        
        values = pd.to_numeric(
            self.data[col_name].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
        valid_mask = ~values.isna()
        
        if method in ['statistical', 'both']:
            z_scores = np.abs(stats.zscore(values[valid_mask]))
            z_outliers = pd.Series(False, index=self.data.index)
            z_outliers[valid_mask] = z_scores > z_threshold
        else:
            z_outliers = pd.Series(False, index=self.data.index)
        
        if method in ['iqr', 'both']:
            Q1 = values[valid_mask].quantile(0.25)
            Q3 = values[valid_mask].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            iqr_outliers = (values < lower_bound) | (values > upper_bound)
        else:
            iqr_outliers = pd.Series(False, index=self.data.index)
        
        if method == 'both':
            outliers = z_outliers | iqr_outliers
        elif method == 'statistical':
            outliers = z_outliers
        else:
            outliers = iqr_outliers
        
        final_mask = valid_mask & ~outliers
        
        artifact_count = (~final_mask).sum()
        artifact_pct = (artifact_count / len(self.data)) * 100
        
        _vprint(f"ÉTAPE 1 - Détection artefacts ({col_name}):", self.verbose, 1)
        _vprint(f"  - Artefacts détectés : {artifact_count}/{len(self.data)} ({artifact_pct:.2f}%)", self.verbose, 2)
        
        return final_mask
    
    def step2_handle_missing_data(self, column, artifact_mask, 
                                 max_gap_ms=100, method='interpolate'):
        """
        ÉTAPE 2 : Gestion des données manquantes
        CORRIGÉ pour échantillonnage irrégulier - utilise les timestamps réels
        """
        if isinstance(column, int):
            col_name = self.data.columns[column]
        else:
            col_name = column
        
        values = pd.to_numeric(
            self.data[col_name].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
        values[~artifact_mask] = np.nan
        
        timestamps = self.data[self.timestamp_col].values
        
        # Créer un DataFrame avec timestamp et valeurs pour interpolation temporelle
        df_temp = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        # Identifier les gaps temporels réels (pas basés sur les indices !)
        df_temp['time_diff'] = df_temp['timestamp'].diff()
        large_gaps = df_temp['time_diff'] > max_gap_ms
        
        if method == 'interpolate':
            # Interpolation linéaire basée sur les timestamps réels
            valid_mask = ~df_temp['value'].isna()
            
            if valid_mask.sum() >= 2:  # Besoin d'au moins 2 points pour interpoler
                # Utiliser interpolation temporelle
                valid_times = df_temp.loc[valid_mask, 'timestamp'].values
                valid_values = df_temp.loc[valid_mask, 'value'].values
                
                # Créer fonction d'interpolation
                interp_func = interp1d(
                    valid_times, 
                    valid_values, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # Interpoler seulement les petits gaps
                processed = df_temp['value'].copy()
                small_gaps = df_temp['value'].isna() & ~large_gaps
                processed[small_gaps] = interp_func(df_temp.loc[small_gaps, 'timestamp'])
                
            else:
                processed = df_temp['value']
                
        elif method == 'forward_fill':
            processed = df_temp['value'].fillna(method='ffill')
            processed[large_gaps] = np.nan
            
        elif method == 'backward_fill':
            processed = df_temp['value'].fillna(method='bfill')
            processed[large_gaps] = np.nan
            
        else:
            processed = df_temp['value']
        
        remaining_nan = processed.isna().sum()
        _vprint(f"\nÉTAPE 2 - Gestion données manquantes ({col_name}):", self.verbose, 1)
        _vprint(f"  - Méthode : {method}", self.verbose, 2)
        _vprint(f"  - Gap max interpolé : {max_gap_ms} ms", self.verbose, 2)
        _vprint(f"  - NaN restants : {remaining_nan} ({remaining_nan/len(self.data)*100:.2f}%)", self.verbose, 2)
        
        return processed.values
    
    def step2_5_resample_if_needed(self, timestamps, values, 
                                   target_interval_ms=None, 
                                   force_resample=False):
        """
        ÉTAPE 2.5 : Rééchantillonnage si nécessaire
        CRUCIAL pour échantillonnage irrégulier avant le lissage
        
        Utilise _create_uniform_timestamps pour garantir que toutes les colonnes
        utilisent le même index temporel uniforme.
        """
        # Vérifier si rééchantillonnage nécessaire
        intervals = np.diff(timestamps)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        if not force_resample and cv < 0.1:
            _vprint(f"\nÉTAPE 2.5 - Rééchantillonnage:", self.verbose, 1)
            _vprint(f"  - Échantillonnage déjà régulier (CV={cv:.2%})", self.verbose, 2)
            _vprint(f"  - Pas de rééchantillonnage nécessaire", self.verbose, 2)
            # Créer quand même l'index uniforme pour cohérence
            uniform_timestamps = self._create_uniform_timestamps(target_interval_ms)
            return uniform_timestamps, values
        
        # Créer ou réutiliser l'index temporel uniforme
        uniform_timestamps = self._create_uniform_timestamps(target_interval_ms)
        
        # Interpoler les valeurs sur le nouvel index
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() >= 2:
            interp_func = interp1d(
                timestamps[valid_mask],
                values[valid_mask],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            resampled_values = interp_func(uniform_timestamps)
        else:
            resampled_values = np.full(len(uniform_timestamps), np.nan)
        
        _vprint(f"\nÉTAPE 2.5 - Rééchantillonnage:", self.verbose, 1)
        _vprint(f"  - Points originaux : {len(timestamps)}", self.verbose, 2)
        _vprint(f"  - Points rééchantillonnés : {len(uniform_timestamps)}", self.verbose, 2)
        _vprint(f"  - Intervalle uniforme : {self.target_interval_ms:.2f} ms", self.verbose, 2)
        
        return uniform_timestamps, resampled_values
    
    def step3_smoothing(self, timestamps, values, method='savgol', 
                       window_size=5, poly_order=2, resample_first=True):
        """
        ÉTAPE 3 : Lissage des données
        CORRIGÉ : rééchantillonne d'abord si échantillonnage irrégulier
        """
        if method == 'none':
            return values
        
        # IMPORTANT : Rééchantillonner d'abord si irrégulier
        if resample_first:
            timestamps, values = self.step2_5_resample_if_needed(
                timestamps, values, force_resample=False
            )
        
        valid_mask = ~np.isnan(values)
        
        if method == 'savgol':
            if window_size % 2 == 0:
                window_size += 1
            smoothed = values.copy()
            if valid_mask.sum() >= window_size:
                smoothed[valid_mask] = savgol_filter(
                    values[valid_mask], 
                    window_size, 
                    poly_order
                )
            else:
                _vprint(f"  ⚠️ Pas assez de points pour Savitzky-Golay (besoin {window_size}, a {valid_mask.sum()})", self.verbose, 2)
                
        elif method == 'median':
            smoothed = values.copy()
            if valid_mask.sum() >= window_size:
                smoothed[valid_mask] = medfilt(values[valid_mask], kernel_size=window_size)
            else:
                _vprint(f"  ⚠️ Pas assez de points pour filtre médian", self.verbose, 2)
                
        elif method == 'moving_average':
            # Moyenne mobile sur index régulier (après rééchantillonnage)
            smoothed = pd.Series(values).rolling(
                window=window_size, center=True, min_periods=1
            ).mean().values
            
        else:
            smoothed = values
        
        _vprint(f"\nÉTAPE 3 - Lissage:", self.verbose, 1)
        _vprint(f"  - Méthode : {method}", self.verbose, 2)
        _vprint(f"  - Fenêtre : {window_size}", self.verbose, 2)
        
        return smoothed
    
    def step4_baseline_correction(self, timestamps, values, 
                                 baseline_method='first_n_seconds',
                                 baseline_duration=30, percentile=10):
        """
        ÉTAPE 4 : Correction de baseline
        (Fonctionne avec échantillonnage irrégulier - utilise timestamps réels)
        """
        if baseline_method == 'none':
            return values
        
        valid_mask = ~np.isnan(values)
        
        if baseline_method == 'first_n_seconds':
            baseline_start = timestamps.min()
            baseline_end = baseline_start + (baseline_duration * 1000)
            baseline_mask = (timestamps >= baseline_start) & (timestamps <= baseline_end) & valid_mask
            baseline_value = values[baseline_mask].mean()
            
        elif baseline_method == 'percentile':
            baseline_value = np.nanpercentile(values[valid_mask], percentile)
            
        elif baseline_method == 'mean':
            baseline_value = np.nanmean(values[valid_mask])
        else:
            baseline_value = 0
        
        corrected = values - baseline_value
        
        _vprint(f"\nÉTAPE 4 - Correction baseline:", self.verbose, 1)
        _vprint(f"  - Méthode : {baseline_method}", self.verbose, 2)
        _vprint(f"  - Valeur baseline : {baseline_value:.4f}", self.verbose, 2)
        
        return corrected
    
    def process_column(self, column, config):
        """
        Pipeline complet pour une colonne
        MODIFIÉ pour gérer l'échantillonnage irrégulier
        """
        if isinstance(column, int):
            col_name = self.data.columns[column]
        else:
            col_name = column
        
        _vprint(f"\n{'='*60}", self.verbose, 2)
        _vprint(f"PREPROCESSING : {col_name}", self.verbose, 2)
        _vprint(f"{'='*60}", self.verbose, 2)
        
        timestamps = self.data[self.timestamp_col].values
        
        # Étape 1 : Détection artefacts
        artifact_mask = self.step1_detect_artifacts(
            column, 
            **config.get('artifact_detection', {})
        )
        
        # Étape 2 : Gestion données manquantes
        values = self.step2_handle_missing_data(
            column, 
            artifact_mask,
            **config.get('missing_data', {})
        )
        
        # Étape 2.5 : Rééchantillonnage si nécessaire (pour lissage)
        resample_config = config.get('resampling', {})
        if resample_config.get('enabled', True):  # Par défaut activé pour irrégulier
            timestamps, values = self.step2_5_resample_if_needed(
                timestamps, values,
                target_interval_ms=resample_config.get('target_interval_ms', None),
                force_resample=resample_config.get('force', False)
            )
        
        # Étape 3 : Lissage (maintenant sur données régulières)
        smoothing_config = config.get('smoothing', {})
        values = self.step3_smoothing(
            timestamps, values,
            resample_first=False,  # Déjà rééchantillonné
            **smoothing_config
        )
        
        # Étape 4 : Correction baseline
        values = self.step4_baseline_correction(
            timestamps, values,
            **config.get('baseline_correction', {})
        )
        
        return pd.Series(values, index=range(len(values))), timestamps
    
    def _is_categorical_column(self, column):
        """
        Vérifie si une colonne est catégorielle (ne doit pas être traitée)
        
        Parameters:
        -----------
        column : str ou int
            Nom ou index de la colonne
            
        Returns:
        --------
        bool : True si colonne catégorielle
        """
        if isinstance(column, int):
            col_name = self.data.columns[column]
        else:
            col_name = column
        
        # Colonnes catégorielles typiques dans Tobii
        categorical_keywords = [
            'Eye movement type', 'Validity', 'Project name', 
            'Participant name', 'Recording name', 'Recording date',
            'Export date', 'Sensor', 'Event'
        ]
        
        # Vérifier par nom
        if any(keyword.lower() in col_name.lower() for keyword in categorical_keywords):
            return True
        
        # Vérifier par dtype
        dtype = self.data[col_name].dtype
        if dtype == 'object' or dtype.name == 'category':
            # Vérifier si c'est vraiment catégoriel (peu de valeurs uniques)
            n_unique = self.data[col_name].nunique()
            n_total = len(self.data)
            if n_unique < 20 and n_unique / n_total < 0.1:  # Moins de 20 valeurs uniques et <10% de variabilité
                return True
        
        return False
    
    def _create_uniform_timestamps(self, target_interval_ms=None):
        """
        Crée l'index temporel uniforme UNE FOIS pour toutes les colonnes
        
        Parameters:
        -----------
        target_interval_ms : float, optional
            Intervalle cible en ms. Si None, auto-détecté.
            
        Returns:
        --------
        array : Index temporel uniforme
        """
        if self.uniform_timestamps is not None:
            return self.uniform_timestamps  # Déjà créé
        
        if target_interval_ms is None:
            # Auto-détection depuis les données
            timestamps = self.data[self.timestamp_col].values
            intervals = np.diff(timestamps)
            intervals = intervals[intervals > 0]
            
            # Filtrer les outliers (gaps très longs)
            Q1 = np.percentile(intervals, 25)
            Q3 = np.percentile(intervals, 75)
            IQR = Q3 - Q1
            normal_intervals = intervals[
                (intervals >= Q1 - 1.5*IQR) & 
                (intervals <= Q3 + 1.5*IQR)
            ]
            
            if len(normal_intervals) > 0:
                target_interval_ms = np.median(normal_intervals)
            else:
                target_interval_ms = np.median(intervals)
        
        self.target_interval_ms = target_interval_ms
        
        min_ts = self.data[self.timestamp_col].min()
        max_ts = self.data[self.timestamp_col].max()
        self.uniform_timestamps = np.arange(min_ts, max_ts + target_interval_ms, target_interval_ms)
        
        return self.uniform_timestamps
    
    def process_columns(self, columns, config, skip_categorical=True):
        """
        Traite plusieurs colonnes en une fois avec vérifications de cohérence
        
        Cette fonction garantit que toutes les colonnes traitées ont le même
        échantillonnage temporel après traitement, ce qui est crucial pour les
        analyses multivariées et le machine learning.
        
        Parameters:
        -----------
        columns : list
            Liste de noms de colonnes ou d'indices à traiter
        config : dict
            Configuration de preprocessing (peut être spécifique par colonne ou uniforme)
        skip_categorical : bool
            Si True, ignore les colonnes catégorielles (défaut: True)
            
        Returns:
        --------
        DataFrame : Données préprocessées avec toutes les colonnes
        dict : Rapport de preprocessing avec statistiques
        """
        _vprint(f"\n{'='*80}", self.verbose, 2)
        _vprint(f"PREPROCESSING MULTI-COLONNES", self.verbose, 1)
        _vprint(f"{'='*80}", self.verbose, 2)
        
        # Filtrer les colonnes catégorielles si demandé
        columns_to_process = []
        skipped_columns = []
        
        for col in columns:
            if isinstance(col, int):
                col_name = self.data.columns[col]
            else:
                col_name = col
            
            if skip_categorical and self._is_categorical_column(col_name):
                skipped_columns.append(col_name)
                _vprint(f"⚠️ Colonne catégorielle ignorée : {col_name}", self.verbose, 2)
            else:
                columns_to_process.append(col)
        
        if skipped_columns:
            _vprint(f"\nColonnes catégorielles ignorées ({len(skipped_columns)}): {skipped_columns}", self.verbose, 2)
        
        if not columns_to_process:
            raise ValueError("Aucune colonne numérique à traiter après filtrage des catégorielles")
        
        # Traiter chaque colonne
        processed_series = {}
        processed_timestamps = {}
        resampling_info = {}
        
        for col in columns_to_process:
            if isinstance(col, int):
                col_name = self.data.columns[col]
            else:
                col_name = col
            
            # Utiliser config spécifique si disponible, sinon config général
            col_config = config.get(col_name, config) if isinstance(config, dict) and col_name in config else config
            
            # Traiter la colonne
            processed, timestamps_processed = self.process_column(col, col_config)
            processed_series[col_name] = processed
            processed_timestamps[col_name] = timestamps_processed
            
            # Stocker les infos de rééchantillonnage pour vérification
            if self.uniform_timestamps is not None:
                resampling_info[col_name] = {
                    'length': len(processed),
                    'interval_ms': self.target_interval_ms,
                    'nan_pct': (processed.isna().sum() / len(processed)) * 100
                }
        
        # Créer DataFrame avec toutes les colonnes traitées
        if self.uniform_timestamps is not None:
            # Utiliser l'index temporel uniforme
            processed_df = pd.DataFrame(processed_series, index=self.uniform_timestamps)
            processed_df.index.name = self.timestamp_col
            processed_df[self.timestamp_col] = self.uniform_timestamps
        else:
            # Utiliser l'index original
            first_col = list(processed_timestamps.keys())[0]
            first_timestamps = processed_timestamps[first_col]
            
            # Vérifier si toutes les colonnes ont les mêmes timestamps
            all_same = all(np.array_equal(processed_timestamps[first_col], ts) 
                        for ts in processed_timestamps.values())
            
            if all_same:
                # Toutes les colonnes ont les mêmes timestamps
                processed_df = pd.DataFrame(processed_series)
                processed_df[self.timestamp_col] = first_timestamps
                processed_df.index = range(len(processed_df))
            else:
                # Les timestamps diffèrent (ne devrait pas arriver sans resampling)
                _vprint("⚠️ ATTENTION : Les timestamps diffèrent entre colonnes", self.verbose, 1)
                # Utiliser les timestamps de la première colonne
                processed_df = pd.DataFrame(processed_series)
                processed_df[self.timestamp_col] = first_timestamps
                processed_df.index = range(len(processed_df))
        # Vérifier la cohérence de l'échantillonnage
        self._verify_sampling_consistency(resampling_info)
        
        # Générer rapport
        report = self._generate_report(processed_df, resampling_info, skipped_columns)
        
        self.processed_data = processed_df
        self.preprocessing_report = report
        
        return processed_df, report
    
    def _verify_sampling_consistency(self, resampling_info):
        """
        Vérifie que toutes les colonnes ont le même échantillonnage après traitement
        
        Parameters:
        -----------
        resampling_info : dict
            Informations de rééchantillonnage par colonne
        """
        if not resampling_info:
            return
        
        _vprint(f"\n{'='*80}", self.verbose, 2)
        _vprint(f"VÉRIFICATION DE COHÉRENCE DE L'ÉCHANTILLONNAGE", self.verbose, 1)
        _vprint(f"{'='*80}", self.verbose, 2)
        
        lengths = [info['length'] for info in resampling_info.values()]
        intervals = [info['interval_ms'] for info in resampling_info.values()]
        
        # Vérifier longueurs identiques
        if len(set(lengths)) > 1:
            _vprint(f"⚠️ ERREUR : Longueurs différentes détectées!", self.verbose, 2)
            for col_name, info in resampling_info.items():
                _vprint(f"  - {col_name}: {info['length']} points", self.verbose, 2)
            raise ValueError("Les colonnes traitées n'ont pas la même longueur après preprocessing")
        else:
            _vprint(f"✓ Toutes les colonnes ont la même longueur : {lengths[0]} points", self.verbose, 2)
        
        # Vérifier intervalles identiques
        if len(set(intervals)) > 1:
            _vprint(f"⚠️ ERREUR : Intervalles différents détectés!", self.verbose, 2)
            for col_name, info in resampling_info.items():
                _vprint(f"  - {col_name}: {info['interval_ms']:.2f} ms", self.verbose, 2)
            raise ValueError("Les colonnes traitées n'ont pas le même intervalle d'échantillonnage")
        else:
            _vprint(f"✓ Toutes les colonnes ont le même intervalle : {intervals[0]:.2f} ms", self.verbose, 2)
        
        _vprint(f"✓ Cohérence de l'échantillonnage vérifiée avec succès", self.verbose, 2)
    
    def _generate_report(self, processed_df, resampling_info, skipped_columns):
        """
        Génère un rapport détaillé du preprocessing
        
        Parameters:
        -----------
        processed_df : DataFrame
            Données préprocessées
        resampling_info : dict
            Informations de rééchantillonnage
        skipped_columns : list
            Colonnes ignorées
            
        Returns:
        --------
        dict : Rapport de preprocessing
        """
        report = {
            'columns_processed': list(processed_df.columns),
            'columns_skipped': skipped_columns,
            'n_columns_processed': len(processed_df.columns),
            'n_columns_skipped': len(skipped_columns),
            'n_samples': len(processed_df),
            'resampling': {},
            'nan_statistics': {},
            'summary': {}
        }
        
        # Infos de rééchantillonnage
        if resampling_info:
            first_col = list(resampling_info.keys())[0]
            report['resampling'] = {
                'target_interval_ms': resampling_info[first_col]['interval_ms'],
                'sampling_rate_hz': 1000 / resampling_info[first_col]['interval_ms'],
                'n_samples': resampling_info[first_col]['length']
            }
        
        # Statistiques NaN par colonne
        for col in processed_df.columns:
            nan_count = processed_df[col].isna().sum()
            nan_pct = (nan_count / len(processed_df)) * 100
            report['nan_statistics'][col] = {
                'nan_count': int(nan_count),
                'nan_percentage': float(nan_pct),
                'valid_count': int(len(processed_df) - nan_count),
                'valid_percentage': float(100 - nan_pct)
            }
        
        # Résumé global
        all_nan_pcts = [stats['nan_percentage'] for stats in report['nan_statistics'].values()]
        report['summary'] = {
            'mean_nan_percentage': float(np.mean(all_nan_pcts)),
            'median_nan_percentage': float(np.median(all_nan_pcts)),
            'min_nan_percentage': float(np.min(all_nan_pcts)),
            'max_nan_percentage': float(np.max(all_nan_pcts)),
            'total_n_samples': int(len(processed_df)),
            'total_n_columns': int(len(processed_df.columns))
        }
        
        # Afficher le rapport
        _vprint(f"\n{'='*80}", self.verbose, 2)
        _vprint(f"RAPPORT DE PREPROCESSING", self.verbose, 1)
        _vprint(f"{'='*80}", self.verbose, 2)
        _vprint(f"\nColonnes traitées : {report['n_columns_processed']}", self.verbose, 2)
        _vprint(f"Colonnes ignorées : {report['n_columns_skipped']}", self.verbose, 2)
        _vprint(f"Nombre d'échantillons : {report['n_samples']}", self.verbose, 2)
        
        if report['resampling']:
            _vprint(f"\nRééchantillonnage :", self.verbose, 2)
            _vprint(f"  - Intervalle : {report['resampling']['target_interval_ms']:.2f} ms", self.verbose, 2)
            _vprint(f"  - Fréquence : {report['resampling']['sampling_rate_hz']:.2f} Hz", self.verbose, 2)
        
        _vprint(f"\nStatistiques NaN par colonne :", self.verbose, 2)
        for col, stats in report['nan_statistics'].items():
            _vprint(f"  - {col}: {stats['nan_percentage']:.2f}% NaN ({stats['valid_count']} valeurs valides)", self.verbose, 2)
        
        _vprint(f"\nRésumé global :", self.verbose, 2)
        _vprint(f"  - NaN moyen : {report['summary']['mean_nan_percentage']:.2f}%", self.verbose, 2)
        _vprint(f"  - NaN médian : {report['summary']['median_nan_percentage']:.2f}%", self.verbose, 2)
        _vprint(f"  - NaN min : {report['summary']['min_nan_percentage']:.2f}%", self.verbose, 2)
        _vprint(f"  - NaN max : {report['summary']['max_nan_percentage']:.2f}%", self.verbose, 2)
        
        return report

# ============================================================================
# CLASS PROCESSING TOBII
# ============================================================================

class ProcessingTobii:
    """
    Classe pour le processing des données Tobii (regroupe les classes FirstPreprocessing et TobiiTimeSeriesPreprocessor)
    """
    def __init__(self, tobii_data: pd.DataFrame, params_dict: dict, columns_to_process: list[str] = None, verbose: int = 2):
        self.tobii_data = tobii_data.copy()
        self.params_dict = params_dict
        self.columns_to_process =  columns_to_process
        self.verbose = verbose
        
    def apply_processing_steps(self):
        first_prep = FirstPreprocessing(self.tobii_data, verbose=self.verbose)
        tobii_cleaned = first_prep.apply_all_steps()
        final_prep = TobiiTimeSeriesPreprocessor(tobii_cleaned, verbose=self.verbose)
        tobii_processed, report = final_prep.process_columns( columns=self.columns_to_process, config = self.params_dict, skip_categorical=True)
        

        return tobii_processed, report


# ============================================================================
# Paramètres par défaut
# ============================================================================



columns_to_process = [
                'Pupil diameter left',
                'Pupil diameter right',
                'Gaze point X',
                'Gaze point Y',
                'Gyro X',
                'Gyro Y',
                'Gyro Z',
                'Accelerometer X',
                'Accelerometer Y',
                'Accelerometer Z'
            ]

params_dict = {
    'artifact_detection': {
        'method': 'both',
        'z_threshold': 4, #3 dans le papier
        'iqr_factor': 2
    },
    'resampling': {
        'enabled': False,  # ⚠️ DÉSACTIVÉ - on garde les timestamps originaux
        'target_interval_ms': None,
        'force': False
    },
    'missing_data': {
        'max_gap_ms': 10000,  # Adaptatif selon analyse préalable
        'method': 'interpolate'  # Interpolation sur timestamps réels
    },
    'smoothing': {
        'method': 'median',  # Moyenne mobile adaptée aux données irrégulières
        'window_size': 3,
        'poly_order': 2  # Ignoré pour moving_average
    },
    'baseline_correction': {
        'baseline_method': 'None' ,#'first_n_seconds', None parait mieux pour la reduction du bruit
        'baseline_duration': 30
    }
}