# Pipeline d'analyse Eye-Tracking - Etude SDS2

Ce projet analyse les mouvements oculaires (eye-tracking) de participants effectuant une tache de construction LEGO a partir d'un plan. L'objectif est de comparer les patterns visuels entre patients atteints de maladies neurologiques et sujets controles sains.

## Table des matieres

1. [Qu'est-ce que l'eye-tracking ?](#quest-ce-que-leye-tracking-)
2. [Contexte de l'etude](#contexte-de-letude)
3. [Concepts cles](#concepts-cles)
4. [Architecture du projet](#architecture-du-projet)
5. [Installation](#installation)
6. [Structure des donnees](#structure-des-donnees)
7. [Guide d'utilisation](#guide-dutilisation)
8. [Metriques expliquees](#metriques-expliquees)
9. [FAQ](#faq)

---

## Qu'est-ce que l'eye-tracking ?

L'eye-tracking (oculometrie) est une technique qui permet de mesurer precisement ou une personne regarde. Un appareil special (ici, un Tobii) projette une lumiere infrarouge vers les yeux et detecte les reflets sur la cornee et la pupille. Cela permet de savoir :

- **Ou** la personne regarde sur l'ecran (position X, Y en pixels)
- **Combien de temps** elle regarde chaque zone
- **Comment** ses yeux se deplacent (mouvements rapides vs lents)
- **La taille de ses pupilles** (qui varie avec l'effort mental)

### Pourquoi c'est utile en recherche medicale ?

Les mouvements oculaires sont controles par le cerveau. Quand certaines zones cerebrales sont affectees par une maladie neurologique, les patterns de regard peuvent changer de facon mesurable. C'est une fenetre non-invasive sur le fonctionnement cerebral.

---

## Contexte de l'etude

### L'etude SDS2

Cette etude longitudinale suit des participants sur 36 mois avec des visites regulieres :
- **M0** : Debut de l'etude (baseline)
- **M12** : 12 mois plus tard
- **M24** : 24 mois plus tard
- **M36** : 36 mois plus tard

A chaque visite, les participants portent un eye-tracker Tobii pendant qu'ils realisent une tache.

### La tache LEGO

Les participants doivent construire une structure avec des briques LEGO en suivant un plan de construction affiche a l'ecran. Cette tache permet d'observer :
- Comment ils consultent le plan (ou regardent-ils ?)
- Comment ils organisent leur regard entre le plan et les pieces
- Combien de temps ils fixent chaque element
- Comment leur attention evolue pendant la tache

### Les deux groupes

- **Patients (P)** : Personnes atteintes de maladies neurologiques
- **Controles (C)** : Personnes saines, appariees en age et sexe

Comparer ces deux groupes permet d'identifier les differences de comportement visuel liees a la maladie.

---

## Concepts cles

### Les mouvements oculaires

L'oeil ne se deplace pas de facon continue et fluide. Il alterne entre :

#### Fixations
- **Definition** : Moments ou l'oeil reste relativement stable sur une zone
- **Duree typique** : 150-500 millisecondes
- **Signification** : C'est pendant les fixations que le cerveau traite l'information visuelle
- **Analogie** : Comme poser le doigt sur une carte pour lire un lieu

#### Saccades
- **Definition** : Mouvements rapides de l'oeil d'un point a un autre
- **Duree typique** : 20-80 millisecondes
- **Vitesse** : Jusqu'a 500 degres par seconde
- **Signification** : L'oeil "saute" entre les zones d'interet ; on ne voit quasiment rien pendant une saccade
- **Analogie** : Comme deplacer rapidement le doigt vers une autre zone de la carte

#### Clignements (Blinks)
- **Definition** : Fermeture temporaire des paupieres
- **Duree typique** : 100-400 millisecondes
- **Signification** : Naturels et necessaires, mais peuvent aussi augmenter avec la fatigue

### Le diametre pupillaire

La pupille (le rond noir au centre de l'oeil) change de taille selon :
- **La luminosite** : Se contracte quand il y a plus de lumiere
- **L'effort mental** : Se dilate quand on reflechit intensement
- **Les emotions** : Se dilate face a quelque chose d'interessant ou stressant

Dans ce projet, on mesure la **variabilite pupillaire** (coefficient de variation) comme indicateur de charge cognitive.

### Les donnees BORIS

En parallele de l'eye-tracking, des observateurs codent le comportement du participant avec le logiciel BORIS (Behavioral Observation Research Interactive Software). Ils notent :
- Quand le participant regarde le plan
- Quand il manipule les pieces
- Quand il hesite ou fait une pause
- Les erreurs commises

Cela permet de croiser les donnees visuelles (eye-tracking) avec les actions comportementales.

---

## Architecture du projet

```
projet_medecin/
│
├── src/                          # Code source Python
│   ├── tobii_pipeline/           # Traitement des donnees eye-tracking
│   │   ├── loader.py             # Chargement des fichiers TSV
│   │   ├── cleaner.py            # Nettoyage des donnees
│   │   ├── postprocess.py        # Pre-traitement avance
│   │   └── analysis/             # Calcul des metriques et visualisations
│   │
│   ├── boris_pipeline/           # Traitement des donnees comportementales
│   │   ├── loader.py             # Chargement des fichiers Excel
│   │   └── analysis/             # Analyse des comportements
│   │
│   └── integration/              # Croisement Tobii + BORIS
│       ├── alignment.py          # Alignement temporel
│       └── cross_modal.py        # Analyse croisee
│
├── scripts/                      # Scripts d'analyse
│   ├── run_analysis.py           # Analyse de groupe
│   └── run_patient_analysis.py   # Analyse individuelle
│
├── notebooks/                    # Notebooks Jupyter pedagogiques
│   └── exemple_analyse.ipynb     # Tutoriel pas-a-pas
│
├── Data/                         # Donnees (a ajouter par l'utilisateur)
│   ├── data_G/
│   │   ├── Tobii/                # Fichiers .tsv
│   │   └── Boris/                # Fichiers .xlsx
│   └── data_L/
│       ├── Tobii/
│       └── Boris/
│
└── figures/                      # Graphiques generes
```

### Flux de traitement

```
┌─────────────────┐     ┌─────────────────┐
│  Tobii (.tsv)   │     │  BORIS (.xlsx)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   Chargement    │     │   Chargement    │
│  (loader.py)    │     │  (loader.py)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐              │
│   Nettoyage     │              │
│ - Decimales     │              │
│ - Filtrage      │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Pre-traitement  │              │
│ - Interpolation │              │
│ - Clignements   │              │
│ - Evenements    │              │
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│              Integration                 │
│  Alignement temporel Tobii ↔ BORIS      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│               Analyse                    │
│  - Metriques (fixations, pupilles...)   │
│  - Comparaisons Patient vs Controle     │
│  - Tendances longitudinales             │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│            Visualisations               │
│  - Cartes de chaleur                    │
│  - Graphiques statistiques              │
│  - Figures de publication               │
└─────────────────────────────────────────┘
```

---

## Installation

### Prerequis

- Python 3.10 ou superieur
- pip (gestionnaire de paquets Python)
- Git (optionnel, pour cloner le projet)

### Etapes d'installation

1. **Cloner ou telecharger le projet**
   ```bash
   git clone <url-du-projet>
   cd projet_medecin
   ```

2. **Creer un environnement virtuel** (recommande)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # ou
   .venv\Scripts\activate     # Windows
   ```

3. **Installer les dependances**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Ajouter les donnees**

   Placez le dossier `Data/` a la racine du projet avec la structure suivante :
   ```
   Data/
   ├── data_G/
   │   ├── Tobii/    # Fichiers .tsv de l'eye-tracker
   │   └── Boris/    # Fichiers .xlsx des observations
   └── data_L/
       ├── Tobii/
       └── Boris/
   ```

5. **Verifier l'installation**
   ```bash
   pytest  # Lance les tests
   ```

---

## Structure des donnees

### Fichiers Tobii (eye-tracking)

Les fichiers `.tsv` (Tab-Separated Values) contiennent les mesures brutes du Tobii a 100 Hz (100 mesures par seconde).

**Convention de nommage** : `{ID}_{Participant}_{Etude}_{Groupe}_{Mois}_{Visite}_{Date} Data Export.tsv`

**Exemple** : `G213_FAUJea_SDS2_P_M36_V4_25062025 Data Export.tsv`

| Element | Description | Exemple |
|---------|-------------|---------|
| ID | Identifiant unique | G213 |
| Participant | Code 6 caracteres | FAUJea |
| Etude | Nom de l'etude | SDS2 |
| Groupe | P (Patient) ou C (Controle) | P |
| Mois | Timepoint | M36 |
| Visite | Numero de visite | V4 |
| Date | JJMMAAAA | 25062025 |

**Colonnes importantes** :
- `Recording timestamp` : Temps en microsecondes
- `Gaze point X`, `Gaze point Y` : Position du regard (pixels)
- `Pupil diameter left`, `Pupil diameter right` : Taille des pupilles (mm)
- `Validity left`, `Validity right` : Qualite de la mesure

### Fichiers BORIS (comportement)

Deux types de fichiers Excel :

1. **Fichiers agreges** (`*_agregated.xlsx`) : Evenements individuels avec debut/fin
2. **Fichiers time budget** (`*.xlsx`) : Resume statistique par comportement

---

## Guide d'utilisation

### Analyse d'un participant individuel

```bash
python scripts/run_patient_analysis.py FAUJea
```

Options disponibles :
- `-o results/` : Changer le dossier de sortie
- `--no-plots` : Generer uniquement les metriques (sans graphiques)
- `--nrows 50000` : Limiter le nombre de lignes (pour tester)

**Sorties generees** :
- `metrics/tobii_metrics.csv` : Metriques eye-tracking
- `metrics/boris_metrics.csv` : Metriques comportementales
- `metrics/cross_modal_metrics.csv` : Metriques croisees
- `plots/M0_V1/` : Graphiques par visite

### Analyse de groupe (tous les participants)

```bash
python scripts/run_analysis.py
```

Options :
- `--nrows 100000` : Echantillonner les donnees
- `--no-boris` : Ignorer les donnees BORIS

**Sorties generees** :
- `figures/summary_report.csv` : Toutes les metriques
- `figures/group_statistics.csv` : Tests statistiques
- `figures/figure1_group_comparison.png` : Comparaison Patient vs Controle
- `figures/figure2_longitudinal.png` : Evolution temporelle
- `figures/figure3_behavioral.png` : Analyse comportementale

### Utilisation dans un notebook Jupyter

Voir `notebooks/exemple_analyse.ipynb` pour un tutoriel interactif.

---

## Metriques expliquees

### Metriques de qualite des donnees

| Metrique | Definition | Interpretation |
|----------|------------|----------------|
| **Validity rate** | % d'echantillons ou le regard est detecte | > 80% = bonne qualite |
| **Tracking ratio** | % de donnees eye-tracker vs autres capteurs | Proche de 100% idealement |

### Metriques de regard

| Metrique | Definition | Interpretation |
|----------|------------|----------------|
| **Gaze dispersion** | Ecart-type de la position du regard (pixels) | Eleve = regard eparpille ; Bas = regard concentre |
| **Gaze center** | Position moyenne du regard (x, y) | Ou se concentre l'attention |
| **Quadrant distribution** | % du regard dans chaque quart de l'ecran | Detecte les biais spatiaux |

### Metriques pupillaires

| Metrique | Definition | Interpretation |
|----------|------------|----------------|
| **Pupil mean** | Diametre moyen des pupilles (mm) | Taille typique : 2-8 mm |
| **Pupil variability** | Coefficient de variation (ecart-type / moyenne) | Eleve = charge cognitive fluctuante |

### Metriques de fixations

| Metrique | Definition | Interpretation |
|----------|------------|----------------|
| **Fixation count** | Nombre de fixations detectees | Plus = plus d'exploration visuelle |
| **Fixation mean duration** | Duree moyenne d'une fixation (ms) | Long = traitement approfondi |
| **Fixation rate** | Nombre de fixations par seconde | Rythme d'exploration |
| **Fixation dispersion** | Etalement spatial d'une fixation (degres) | Stabilite du regard pendant fixation |

### Metriques de saccades

| Metrique | Definition | Interpretation |
|----------|------------|----------------|
| **Saccade count** | Nombre de saccades detectees | Mouvements oculaires rapides |
| **Saccade amplitude** | Distance parcourue par l'oeil (degres) | Grande = larges deplacements |
| **Saccade velocity** | Vitesse maximale (degres/seconde) | Anomalies = problemes neurologiques |

### Ce que ces metriques revelent cliniquement

1. **Gaze dispersion elevee chez les patients** : Difficulte a concentrer l'attention
2. **Fixations plus longues** : Traitement visuel plus lent
3. **Moins de fixations** : Exploration visuelle reduite
4. **Pupil variability elevee** : Effort cognitif fluctuant
5. **Saccades anormales** : Problemes de controle moteur oculaire

---

## FAQ

### Les donnees ne se chargent pas

**Probleme** : Message d'erreur "No recordings found"

**Solution** : Verifiez que :
1. Le dossier `Data/` existe a la racine du projet
2. Les sous-dossiers `data_G/Tobii` et `data_L/Tobii` contiennent des fichiers `.tsv`
3. Les noms de fichiers suivent la convention attendue

### Erreur "decimal separator"

**Probleme** : Les nombres ne sont pas reconnus correctement

**Explication** : Le Tobii exporte avec des virgules comme separateurs decimaux (format europeen : `3,14` au lieu de `3.14`). Le pipeline gere automatiquement cette conversion.

**Solution** : Assurez-vous d'utiliser la fonction `clean_recording()` apres le chargement.

### Le traitement est tres long

**Solution** : Utilisez l'option `--nrows` pour limiter le nombre de lignes :
```bash
python scripts/run_analysis.py --nrows 50000
```

### Comment interpreter les p-values ?

Les tests statistiques (Mann-Whitney U) comparent les groupes Patient et Controle :
- `p < 0.05` (*) : Difference significative
- `p < 0.01` (**) : Tres significatif
- `p < 0.001` (***) : Hautement significatif

Un effet size (d de Cohen) complete l'interpretation :
- `|d| < 0.2` : Effet negligeable
- `0.2 <= |d| < 0.5` : Petit effet
- `0.5 <= |d| < 0.8` : Effet moyen
- `|d| >= 0.8` : Grand effet

### Ou trouver de l'aide supplementaire ?

- Consultez le notebook `notebooks/exemple_analyse.ipynb` pour un tutoriel interactif
- Les docstrings dans le code source contiennent des explications detaillees
- Pour les questions techniques : ouvrir une issue sur le depot

---

## Ressources pour aller plus loin

### Eye-tracking
- Holmqvist, K. et al. (2011). *Eye Tracking: A comprehensive guide to methods and measures*
- Duchowski, A. (2017). *Eye Tracking Methodology: Theory and Practice*

### Analyse statistique
- Field, A. (2017). *Discovering Statistics Using IBM SPSS Statistics*

### Python pour la science des donnees
- McKinney, W. (2017). *Python for Data Analysis*
- VanderPlas, J. (2016). *Python Data Science Handbook*

---

## Licence et citation

Ce projet est developpe dans le cadre de l'etude SDS2. Pour toute utilisation des donnees ou du code, veuillez contacter les responsables de l'etude.
