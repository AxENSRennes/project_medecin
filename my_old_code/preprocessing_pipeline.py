import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt

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
    
    def __init__(self, tobii_data, timestamp_col=None):
        """
        Initialise le preprocesseur
        
        Parameters:
        -----------
        tobii_data : DataFrame
            Données Tobii avec colonnes temporelles et mesures
        timestamp_col : str, optional
            Nom de la colonne timestamp (par défaut: première colonne)
        """
        self.data = tobii_data.copy()
        self.timestamp_col = timestamp_col if timestamp_col else tobii_data.columns[0]
        self.processed_data = None
        self.preprocessing_report = {}
        self.uniform_timestamps = None  # Index temporel uniforme (créé une fois)
        self.target_interval_ms = None  # Intervalle cible (déterminé une fois)
        
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
        
        print(f"Analyse de l'échantillonnage:")
        print(f"  - Échantillonnage régulier : {self.is_regular}")
        print(f"  - Intervalle médian : {self.median_interval:.2f} ms")
        print(f"  - Coefficient de variation : {cv:.2%}")
        
        if not self.is_regular:
            print(f"  ⚠️ Échantillonnage IRRÉGULIER détecté - rééchantillonnage recommandé")
    
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
        
        print(f"ÉTAPE 1 - Détection artefacts ({col_name}):")
        print(f"  - Artefacts détectés : {artifact_count}/{len(self.data)} ({artifact_pct:.2f}%)")
        
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
        print(f"\nÉTAPE 2 - Gestion données manquantes ({col_name}):")
        print(f"  - Méthode : {method}")
        print(f"  - Gap max interpolé : {max_gap_ms} ms")
        print(f"  - NaN restants : {remaining_nan} ({remaining_nan/len(self.data)*100:.2f}%)")
        
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
            print(f"\nÉTAPE 2.5 - Rééchantillonnage:")
            print(f"  - Échantillonnage déjà régulier (CV={cv:.2%})")
            print(f"  - Pas de rééchantillonnage nécessaire")
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
        
        print(f"\nÉTAPE 2.5 - Rééchantillonnage:")
        print(f"  - Points originaux : {len(timestamps)}")
        print(f"  - Points rééchantillonnés : {len(uniform_timestamps)}")
        print(f"  - Intervalle uniforme : {self.target_interval_ms:.2f} ms")
        
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
                print(f"  ⚠️ Pas assez de points pour Savitzky-Golay (besoin {window_size}, a {valid_mask.sum()})")
                
        elif method == 'median':
            smoothed = values.copy()
            if valid_mask.sum() >= window_size:
                smoothed[valid_mask] = medfilt(values[valid_mask], kernel_size=window_size)
            else:
                print(f"  ⚠️ Pas assez de points pour filtre médian")
                
        elif method == 'moving_average':
            # Moyenne mobile sur index régulier (après rééchantillonnage)
            smoothed = pd.Series(values).rolling(
                window=window_size, center=True, min_periods=1
            ).mean().values
            
        else:
            smoothed = values
        
        print(f"\nÉTAPE 3 - Lissage:")
        print(f"  - Méthode : {method}")
        print(f"  - Fenêtre : {window_size}")
        
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
        
        print(f"\nÉTAPE 4 - Correction baseline:")
        print(f"  - Méthode : {baseline_method}")
        print(f"  - Valeur baseline : {baseline_value:.4f}")
        
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
        
        print(f"\n{'='*60}")
        print(f"PREPROCESSING : {col_name}")
        print(f"{'='*60}")
        
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
        print(f"\n{'='*80}")
        print(f"PREPROCESSING MULTI-COLONNES")
        print(f"{'='*80}")
        
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
                print(f"⚠️ Colonne catégorielle ignorée : {col_name}")
            else:
                columns_to_process.append(col)
        
        if skipped_columns:
            print(f"\nColonnes catégorielles ignorées ({len(skipped_columns)}): {skipped_columns}")
        
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
                print("⚠️ ATTENTION : Les timestamps diffèrent entre colonnes")
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
        
        print(f"\n{'='*80}")
        print(f"VÉRIFICATION DE COHÉRENCE DE L'ÉCHANTILLONNAGE")
        print(f"{'='*80}")
        
        lengths = [info['length'] for info in resampling_info.values()]
        intervals = [info['interval_ms'] for info in resampling_info.values()]
        
        # Vérifier longueurs identiques
        if len(set(lengths)) > 1:
            print(f"⚠️ ERREUR : Longueurs différentes détectées!")
            for col_name, info in resampling_info.items():
                print(f"  - {col_name}: {info['length']} points")
            raise ValueError("Les colonnes traitées n'ont pas la même longueur après preprocessing")
        else:
            print(f"✓ Toutes les colonnes ont la même longueur : {lengths[0]} points")
        
        # Vérifier intervalles identiques
        if len(set(intervals)) > 1:
            print(f"⚠️ ERREUR : Intervalles différents détectés!")
            for col_name, info in resampling_info.items():
                print(f"  - {col_name}: {info['interval_ms']:.2f} ms")
            raise ValueError("Les colonnes traitées n'ont pas le même intervalle d'échantillonnage")
        else:
            print(f"✓ Toutes les colonnes ont le même intervalle : {intervals[0]:.2f} ms")
        
        print(f"✓ Cohérence de l'échantillonnage vérifiée avec succès")
    
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
        print(f"\n{'='*80}")
        print(f"RAPPORT DE PREPROCESSING")
        print(f"{'='*80}")
        print(f"\nColonnes traitées : {report['n_columns_processed']}")
        print(f"Colonnes ignorées : {report['n_columns_skipped']}")
        print(f"Nombre d'échantillons : {report['n_samples']}")
        
        if report['resampling']:
            print(f"\nRééchantillonnage :")
            print(f"  - Intervalle : {report['resampling']['target_interval_ms']:.2f} ms")
            print(f"  - Fréquence : {report['resampling']['sampling_rate_hz']:.2f} Hz")
        
        print(f"\nStatistiques NaN par colonne :")
        for col, stats in report['nan_statistics'].items():
            print(f"  - {col}: {stats['nan_percentage']:.2f}% NaN ({stats['valid_count']} valeurs valides)")
        
        print(f"\nRésumé global :")
        print(f"  - NaN moyen : {report['summary']['mean_nan_percentage']:.2f}%")
        print(f"  - NaN médian : {report['summary']['median_nan_percentage']:.2f}%")
        print(f"  - NaN min : {report['summary']['min_nan_percentage']:.2f}%")
        print(f"  - NaN max : {report['summary']['max_nan_percentage']:.2f}%")
        
        return report

# ============================================================================
# CONFIGURATIONS MISE À JOUR POUR ÉCHANTILLONNAGE IRRÉGULIER
# ============================================================================

# CONFIGURATION SANS RESAMPLING - Données irrégulières conservées
COMMON_PARAMS_NO_RESAMPLE = {
    'artifact_detection': {
        'method': 'both',
        'z_threshold': 3,
        'iqr_factor': 1.5
    },
    'resampling': {
        'enabled': False,  # ⚠️ DÉSACTIVÉ - on garde les timestamps originaux
        'target_interval_ms': None,
        'force': False
    },
    'missing_data': {
        'max_gap_ms': None,  # Adaptatif selon analyse préalable
        'method': 'interpolate'  # Interpolation sur timestamps réels
    }
}

# PARAMÈTRES SPÉCIFIQUES SANS RESAMPLING
# Note: Savitzky-Golay nécessite un échantillonnage régulier
# Pour données irrégulières, on utilise des méthodes adaptées

PUPIL_SPECIFIC_NO_RESAMPLE = {
    'smoothing': {
        'method': 'moving_average',  # Moyenne mobile adaptée aux données irrégulières
        'window_size': 7,
        'poly_order': 2  # Ignoré pour moving_average
    },
    'baseline_correction': {
        'baseline_method': 'first_n_seconds',
        'baseline_duration': 30
    }
}

GAZE_SPECIFIC_NO_RESAMPLE = {
    'smoothing': {
        'method': 'moving_average',  # Ou 'none' pour préserver les saccades
        'window_size': 3,
        'poly_order': 2
    },
    'baseline_correction': {
        'baseline_method': 'none'
    }
}

IMU_SPECIFIC_NO_RESAMPLE = {
    'smoothing': {
        'method': 'moving_average',  # Ou 'median' si adapté aux irréguliers
        'window_size': 5
    },
    'baseline_correction': {
        'baseline_method': 'mean'
    }
}

# CONFIGURATIONS FINALES SANS RESAMPLING
PUPIL_CONFIG_NO_RESAMPLE = {**COMMON_PARAMS_NO_RESAMPLE, **PUPIL_SPECIFIC_NO_RESAMPLE}
GAZE_CONFIG_NO_RESAMPLE = {**COMMON_PARAMS_NO_RESAMPLE, **GAZE_SPECIFIC_NO_RESAMPLE}
IMU_CONFIG_NO_RESAMPLE = {**COMMON_PARAMS_NO_RESAMPLE, **IMU_SPECIFIC_NO_RESAMPLE}

# OU configuration uniforme sans resampling
UNIFORM_CONFIG_NO_RESAMPLE = {
    'artifact_detection': {
        'method': 'both',
        'z_threshold': 3,
        'iqr_factor': 1.5
    },
    'resampling': {
        'enabled': False  # ⚠️ DÉSACTIVÉ
    },
    'missing_data': {
        'max_gap_ms': None,
        'method': 'interpolate'
    },
    'smoothing': {
        'method': 'moving_average',  # Adapté aux données irrégulières
        'window_size': 5
    },
    'baseline_correction': {
        'baseline_method': 'none'  # Ou 'mean', 'first_n_seconds', etc.
    }
}



# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

"""
Exemple d'utilisation du pipeline pour traiter plusieurs colonnes :

# Initialiser le preprocesseur
preprocessor = TobiiTimeSeriesPreprocessor(tobii_cleaned)

# Définir les colonnes à traiter
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

# Configuration uniforme (même traitement pour toutes)
processed_df, report = preprocessor.process_columns(
    columns=columns_to_process,
    config=COMMON_PARAMS,  # Ou PUPIL_CONFIG, GAZE_CONFIG, etc.
    skip_categorical=True  # Ignore automatiquement les colonnes catégorielles
)

# Ou configuration spécifique par type de colonne
config_dict = {
    'Pupil diameter left': PUPIL_CONFIG,
    'Pupil diameter right': PUPIL_CONFIG,
    'Gaze point X': GAZE_CONFIG,
    'Gaze point Y': GAZE_CONFIG,
    'Gyro X': IMU_CONFIG,
    'Gyro Y': IMU_CONFIG,
    'Gyro Z': IMU_CONFIG,
    'Accelerometer X': IMU_CONFIG,
    'Accelerometer Y': IMU_CONFIG,
    'Accelerometer Z': IMU_CONFIG
}

processed_df, report = preprocessor.process_columns(
    columns=columns_to_process,
    config=config_dict,
    skip_categorical=True
)

# Le DataFrame retourné a toutes les colonnes avec le même échantillonnage
# Vérifications automatiques :
# - Même longueur pour toutes les colonnes
# - Même intervalle d'échantillonnage
# - Rapport détaillé avec statistiques NaN

# Accéder aux données préprocessées
print(processed_df.head())
print(f"Pourcentage moyen de NaN : {report['summary']['mean_nan_percentage']:.2f}%")
"""