import pandas as pd
import numpy as np
from typing import List, Union


class FirstPreprocessing:
    """
    Classe pour le preprocessing préliminaire des données Tobii
    
    Cette classe effectue les étapes de nettoyage initiales :
    1. Suppression de la première ligne et de colonnes non pertinentes
    2. Réduction à 1 ligne sur 3 (agrégation)
    3. Filtrage par validité des yeux
    """
    
    def __init__(self, tobii_data: pd.DataFrame):
        """
        Initialise le preprocesseur
        
        Parameters:
        -----------
        tobii_data : DataFrame
            Données Tobii brutes
        """
        self.data = tobii_data.copy()
        self.timestamp_col = 'Recording timestamp'
    
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
        
        print(f"ÉTAPE 1 - Suppression première ligne et colonnes")
        print(f"  - Lignes avant : {len(self.data)}")
        print(f"  - Colonnes avant : {len(self.data.columns)}")
        print(f"  - Colonnes à supprimer : {len(columns_to_remove)}")
        
        # Supprimer la première ligne
        self.data = self.data.iloc[1:].reset_index(drop=True)
        
        # Supprimer les colonnes spécifiées
        columns_to_drop = [self.data.columns[i] for i in columns_to_remove if i < len(self.data.columns)]
        
        # Ne pas supprimer Sensor si demandé
        if keep_sensor and 'Sensor' in columns_to_drop:
            columns_to_drop.remove('Sensor')
            print(f"  - Colonne Sensor conservée (nécessaire pour step3)")
        
        self.data = self.data.drop(columns=columns_to_drop)
        
        print(f"  - Lignes après : {len(self.data)}")
        print(f"  - Colonnes après : {len(self.data.columns)}")
        print(f"  - Colonnes supprimées : {columns_to_drop}")
        
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
        print(f"\nÉTAPE 1.5 - Conversion des types de données")
        print(f"  - Précision float : {float_precision}")
        
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
                    print(f"  - {col} → Int64")
                except Exception as e:
                    print(f"  ⚠️ Impossible de convertir {col} en int : {e}")
        
        # Convertir colonnes catégorielles
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('category')
                print(f"  - {col} → category")
        
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
        
        print(f"  - Colonnes numériques converties : {converted_count}")
        
        # Mémoire après
        memory_after = self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_saved = memory_before - memory_after
        
        print(f"  - Mémoire avant : {memory_before:.2f} MB")
        print(f"  - Mémoire après : {memory_after:.2f} MB")
        print(f"  - Mémoire économisée : {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}%)")
        
        return self.data
    
    def step2_reduce_to_one_in_three(self):
        """
        ÉTAPE 2 (FALLBACK) : Réduit à 1 ligne sur 3 avec méthode modulo 3
        
        Méthode simple basée sur la position (modulo 3) - utilisé si Sensor n'est pas disponible
        
        Returns:
        --------
        DataFrame : Données réduites
        """
        print(f"\nÉTAPE 2 (FALLBACK) - Réduction modulo 3")
        print(f"  - Lignes avant : {len(self.data)}")
        
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
        
        print(f"  - Lignes après : {len(self.data)}")
        if n_groups > 0:
            print(f"  - Réduction : {len(self.data) / (n_groups * 3 + remainder) * 100:.1f}%")
        
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
        print(f"\nÉTAPE 2 - Réduction avec colonne Sensor")
        print(f"  - Lignes avant : {len(self.data)}")
        print(f"  - Colonnes disponibles : {list(self.data.columns)}")
        
        if sensor_col not in self.data.columns:
            print(f"  ⚠️ Colonne {sensor_col} non trouvée dans les colonnes disponibles")
            print(f"  ⚠️ Utilisation méthode modulo 3 (fallback)")
            return self.step2_reduce_to_one_in_three()  # Fallback
        
        # Vérifier les valeurs uniques de Sensor
        sensor_values = self.data[sensor_col].value_counts()
        print(f"  - Valeurs Sensor détectées : {sensor_values.to_dict()}")
        print(f"  - Nombre de valeurs uniques : {len(sensor_values)}")
        
        # Identifier les 3 types de capteurs depuis value_counts
        sensor_types = sensor_values.index.tolist()
        if len(sensor_types) != 3:
            print(f"  ⚠️ ERREUR : Attendu 3 types de capteurs, trouvé {len(sensor_types)}")
            print(f"  ⚠️ Types trouvés : {sensor_types}")
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
            print(f"  ⚠️ Impossible d'identifier les 3 capteurs automatiquement")
            print(f"  - Eye tracker : {eye_tracker_type}")
            print(f"  - Gyroscope : {gyro_type}")
            print(f"  - Accelerometer : {acc_type}")
            # Utiliser l'ordre des value_counts comme fallback
            eye_tracker_type = sensor_types[0]
            gyro_type = sensor_types[1]
            acc_type = sensor_types[2]
            print(f"  - Utilisation ordre value_counts : {eye_tracker_type}, {gyro_type}, {acc_type}")
        
        print(f"  - Eye tracker identifié : {eye_tracker_type}")
        print(f"  - Gyroscope identifié : {gyro_type}")
        print(f"  - Accelerometer identifié : {acc_type}")
        
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
            print(f"\n  ⚠️ ERREUR : Aucun groupe de 3 capteurs trouvé !")
            print(f"  - Lignes parcourues : {i}")
            print(f"  - Lignes sautées : {skipped_lines}")
            print(f"  - Valeurs Sensor uniques : {sensor_values.index.tolist()}")
            print(f"  - Utilisation méthode modulo 3 (fallback)")
            return self.step2_reduce_to_one_in_three()  # Fallback
        
        # Créer nouveau DataFrame
        self.data = pd.DataFrame(reduced_rows)
        
        print(f"  - Groupes de 3 capteurs trouvés : {groups_found}")
        print(f"  - Lignes après : {len(self.data)}")
        if len(reduced_rows) > 0:
            reduction_pct = len(self.data) / (len(self.data) * 3) * 100
            print(f"  - Réduction : ~{reduction_pct:.1f}% (1 ligne sur 3)")
        
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
        print(f"\nÉTAPE 3 - Filtrage par validité des yeux")
        print(f"  - Lignes avant : {len(self.data)}")
        
        if validity_left_col not in self.data.columns or \
           validity_right_col not in self.data.columns:
            print(f"  ⚠️ Colonnes de validité non trouvées - pas de filtrage")
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
            print(f"  - Utilisation colonne {sensor_col} pour supprimer groupes complets")
            
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
                print(f"  ⚠️ Nombre de types Sensor incorrect : {len(sensor_types)}")
                print(f"  ⚠️ Utilisation filtrage simple")
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
                
                print(f"  - Types Sensor : Eye tracker={eye_tracker_type}, Gyro={gyro_type}, Acc={acc_type}")
                
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
                
                print(f"  - Lignes Eye tracker invalides : {len(invalid_eye_tracker_indices)}")
                print(f"  - Groupes complets à supprimer : {len(invalid_eye_tracker_indices)}")
                print(f"  - Total lignes à supprimer : {len(indices_to_remove)}")
                
                # Créer masque final
                valid_mask = ~self.data.index.isin(indices_to_remove)
            
            print(f"  - Lignes Eye tracker invalides : {len(invalid_eye_tracker_indices)}")
            print(f"  - Groupes complets à supprimer : {len(invalid_eye_tracker_indices)}")
            print(f"  - Total lignes à supprimer : {len(indices_to_remove)}")
            
        else:
            # Pas de Sensor : filtrage simple (ancienne méthode)
            print(f"  ⚠️ Colonne {sensor_col} non trouvée - filtrage simple")
            valid_mask = ~eye_tracker_invalid
        
        n_before = len(self.data)
        self.data = self.data[valid_mask].reset_index(drop=True)
        n_after = len(self.data)
        
        n_removed = n_before - n_after
        removal_pct = (n_removed / n_before) * 100 if n_before > 0 else 0
        
        print(f"  - Critère : {criterion}")
        print(f"  - Lignes après : {len(self.data)}")
        print(f"  - Lignes supprimées : {n_removed} ({removal_pct:.2f}%)")
        
        if removal_pct > 50:
            print(f"  ⚠️ ATTENTION : Plus de 50% des données supprimées")
        
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
        print(f"\nÉTAPE 4 - Remplacement NaN Fixation point par 0")
        
        nan_replaced_x = 0
        nan_replaced_y = 0
        
        if fixation_x_col in self.data.columns:
            nan_count_x = self.data[fixation_x_col].isna().sum()
            self.data[fixation_x_col] = self.data[fixation_x_col].fillna(0)
            nan_replaced_x = nan_count_x
            print(f"  - {fixation_x_col}: {nan_replaced_x} NaN remplacés par 0")
        else:
            print(f"  ⚠️ Colonne {fixation_x_col} non trouvée")
        
        if fixation_y_col in self.data.columns:
            nan_count_y = self.data[fixation_y_col].isna().sum()
            self.data[fixation_y_col] = self.data[fixation_y_col].fillna(0)
            nan_replaced_y = nan_count_y
            print(f"  - {fixation_y_col}: {nan_replaced_y} NaN remplacés par 0")
        else:
            print(f"  ⚠️ Colonne {fixation_y_col} non trouvée")
        
        total_replaced = nan_replaced_x + nan_replaced_y
        if total_replaced > 0:
            print(f"  - Total NaN remplacés : {total_replaced}")
            print(f"  - Convention : 0 = pas de fixation détectée")
        
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
        print(f"{'='*80}")
        print(f"PREPROCESSING PRÉLIMINAIRE COMPLET")
        print(f"{'='*80}")
        
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
        
        print(f"\n{'='*80}")
        print(f"PREPROCESSING PRÉLIMINAIRE TERMINÉ")
        print(f"{'='*80}")
        print(f"  - Lignes finales : {len(self.data)}")
        print(f"  - Colonnes finales : {len(self.data.columns)}")
        
        return self.data
# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

"""
Exemple d'utilisation :

from first_preprocessing import FirstPreprocessing

# Charger les données Tobii
tobii = pd.read_csv("fichier.tsv", sep="\t")

# Créer le preprocesseur
preprocessor = FirstPreprocessing(tobii)

# Appliquer toutes les étapes
tobii_cleaned = preprocessor.apply_all_steps(
    columns_to_remove=None,  # Utilise les colonnes par défaut
    require_both_valid=False  # Garde si au moins un œil valide
)

# Ou appliquer étape par étape
preprocessor = FirstPreprocessing(tobii)
preprocessor.step1_remove_first_row_and_columns()
preprocessor.step2_reduce_to_one_in_three()
preprocessor.step3_filter_invalid_eyes(require_both_valid=False)
tobii_cleaned = preprocessor.data
"""

