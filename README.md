# Projet NASA : priorisation des géocroiseurs pour la PDCO

**Client :** NASA/JPL Planetary Defense Coordination Office.
**Objectif :** classer les géocroiseurs pour prioriser les ré-observations télescopiques, avec une politique « zéro faux négatif ».

## Structure du dépôt

```
data-mining-202660-team4/
├── nasa-project/            # projet capstone (client NASA PDCO)
│   ├── data/raw/            # CSV Kaggle (non versionnés)
│   ├── notebooks/           # un notebook par étape
│   ├── reports/             # figures et tableaux générés (versionnés)
│   └── src/                 # scripts utilitaires (téléchargement)
├── activities/              # activités hebdomadaires du cours
├── pyproject.toml           # dépendances du projet
└── uv.lock                  # versions exactes des dépendances
```

## Données

Deux CSV Kaggle (`darkmatternet/nasa-near-earth-asteroids-and-close-approaches`), placés dans `nasa-project/data/raw/` (non versionné) :

- `near_earth_asteroids_2025*.csv` : catalogue de 41 281 objets (éléments orbitaux, H, diamètre, MOID, `pha`) ;
- `asteroid_close_approaches_2015_2035*.csv` : 27 430 approches à moins de 0,1 UA.

## Installation avec uv

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer Python (version 3.14, fixée dans `.python-version`) et les dépendances (`pyproject.toml` et `uv.lock`). Il n'est pas nécessaire d'installer Python à la main : uv le télécharge si besoin.

### 1. Installer uv (une seule fois par machine)

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Ou avec winget : `winget install --id=astral-sh.uv -e`

**macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ou avec Homebrew : `brew install uv`

**Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Fermez puis rouvrez le terminal, et vérifiez l'installation :

```bash
uv --version
```

### 2. Cloner le dépôt et créer l'environnement

Les commandes sont identiques sur les trois systèmes :

```bash
git clone https://github.com/victorcardoso-uam/data-mining-202660-team4.git
cd data-mining-202660-team4
uv sync
```

`uv sync` installe la bonne version de Python, crée le dossier `.venv/` et installe exactement les versions de `uv.lock`, y compris les dépendances de développement comme `ipykernel`.

> Sous Windows, si le dépôt est dans un dossier dont le chemin contient des espaces (OneDrive par exemple), mettez le chemin entre guillemets : `cd "C:\Users\...\data-mining-202660-team4"`.

### 3. Télécharger les données

Depuis la racine du dépôt :

```bash
uv run python nasa-project/src/download_data.py
```

Les CSV sont copiés dans `nasa-project/data/raw/`.

## Lancer les notebooks

Un notebook par étape, dans `nasa-project/notebooks/`. Les figures et tableaux sont écrits dans `nasa-project/reports/<étape>/` (versionné, pour les slides).

| Étape | Notebook | Sorties |
|---|---|---|
| 2. Analyse exploratoire | `nasa-project/notebooks/02_eda.ipynb` | `nasa-project/reports/02_eda/` |
| 3. Preprocessing sans fuite | `nasa-project/notebooks/03_preprocessing.ipynb` | `nasa-project/reports/03_preprocessing/` et `nasa-project/data/processed/03_preprocessing/` |
| 4. Régression du diamètre | `nasa-project/notebooks/04_diameter_regression.ipynb` | `nasa-project/reports/04_diameter/` et `nasa-project/data/processed/04_diameter/` |
| 5. Classification et priorisation | `nasa-project/notebooks/05_classification.ipynb` | `nasa-project/reports/05_classification/` et `nasa-project/data/processed/05_classification/` |
| 6. Clustering des profils de risque | `nasa-project/notebooks/06_clustering.ipynb` | `nasa-project/reports/06_clustering/` et `nasa-project/data/processed/06_clustering/` |
| 7. Politique de priorisation | `nasa-project/notebooks/07_risk_policy.ipynb` | `nasa-project/reports/07_risk_policy/` et `nasa-project/data/processed/07_risk_policy/file_priorite.csv` |

Les notebooks se lancent **dans l'ordre** : à partir de l'étape 3, chaque notebook enregistre dans `nasa-project/data/processed/` (non versionné) les jeux train/test et les préprocesseurs utilisés par les étapes suivantes. Après un `git clone`, exécutez donc les notebooks dans l'ordre (03, puis 04, puis 05…).

### Option A : VS Code (recommandé)

1. Ouvrez le dossier du dépôt dans VS Code (extensions **Python** et **Jupyter** installées).
2. Ouvrez le notebook, cliquez sur **Select Kernel** en haut à droite, puis choisissez l'environnement `.venv` du projet.
3. Cliquez sur **Run All**.

### Option B : Jupyter Lab dans le navigateur

Mêmes commandes sur Windows, macOS et Linux :

```bash
uv run --with jupyter jupyter lab
```

`--with jupyter` utilise Jupyter de façon temporaire, sans l'ajouter aux dépendances du projet.

### Option C : exécuter un notebook sans l'ouvrir

Pratique pour régénérer toutes les figures d'une étape :

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace nasa-project/notebooks/02_eda.ipynb
```

## Activer l'environnement (facultatif)

`uv run <commande>` utilise déjà le bon environnement, sans activation. Si vous préférez l'activer :

| Système | Commande |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Pour le désactiver : `deactivate`.

> Sous Windows, si PowerShell refuse d'exécuter le script d'activation, lancez une fois `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Ajouter une dépendance

```bash
uv add nom-du-paquet            # dépendance du projet
uv add --dev nom-du-paquet      # outil de développement (ipykernel, pytest…)
```

Commitez ensuite `pyproject.toml` et `uv.lock`. Les coéquipiers n'ont plus qu'à lancer `uv sync` après un `git pull`.

Ne commitez jamais le dossier `.venv/` : il est propre à chaque machine et déjà ignoré par `.gitignore`.

## Problèmes fréquents

### Windows et OneDrive : `failed to hardlink file` (os error 396)

Par défaut, uv crée des liens physiques (« hardlinks ») entre son cache et le `.venv`. OneDrive ne les accepte pas, ce qui produit l'erreur « Impossible d'exécuter l'opération de cloud sur un fichier comportant des liaisons permanentes ».

Pour corriger, demandez à uv de copier les fichiers au lieu de les lier. Dans PowerShell :

```powershell
[System.Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
```

Fermez puis rouvrez le terminal (et VS Code), videz le cache abîmé, puis relancez :

```powershell
uv cache clean
uv sync
```

Pour la session en cours uniquement : `$env:UV_LINK_MODE = "copy"`.

Sur macOS et Linux, l'équivalent est `export UV_LINK_MODE=copy`, à ajouter dans `~/.zshrc` ou `~/.bashrc` pour le rendre permanent.

## Constats clés de l'EDA

- `pha` est égal à la règle MOID ≤ 0,05 UA et H ≤ 22 pour 99,9 % des objets. Il faut exclure MOID, H et les colonnes dérivées de H si `pha` est la cible.
- 97 % des diamètres sont calculés à partir de H avec un albédo fixe de 0,14. Seuls 1 246 diamètres sont réellement mesurés.
- La chute des approches après mars 2026 vient de la date d'extraction : les objets pas encore découverts n'ont pas de prévisions.
- 80 % des objets n'ont qu'une seule approche dans la fenêtre 2015–2035.

## Constats clés du preprocessing

- Variables retenues pour la classification : éléments orbitaux `a, e, i, q, ad` et famille `class`. MOID, H, diamètre, `size_category` et albédo sont exclus.
- `data_arc_years`, `condition_code` et `first_obs_year` sont des proxys de la taille (|ρ| avec H jusqu'à 0,77). Ils forment une variante testée séparément.
- Audit de fuite en validation croisée (PR-AUC, hasard = 0,06) : H + MOID = 1,00 (fuite), orbite seule = 0,15, orbite + observation = 0,48.
- Régression du diamètre : 1 244 diamètres mesurés, albédo exclu. Les objets mesurés sont nettement plus gros que ceux à prédire (H médian 19 contre 24).

## Constats clés de la régression du diamètre

- Modèle retenu : Random Forest « guidé par la physique ». Il garde la loi exacte log D = constante − H/5 et n'apprend que le terme d'albédo, à partir de H et de l'orbite.
- Sur le test, par rapport à la formule du catalogue (albédo 0,14) : RMSE de 0,222 à 0,203 dex, et les erreurs d'un facteur supérieur à 2 passent de 20 % à 12 %. Précision équivalente au modèle linéaire, mais la forêt n'extrapole pas de tendance absurde aux petits objets.
- Les objets dont l'orbite s'étend au-delà de 2,17 UA sont plus sombres (albédo ≈ 0,07) : la formule du catalogue sous-estime leur diamètre.
- Intervalle à 90 % : de ×0,52 à ×2,14 autour du diamètre prédit (couverture de 88 % sur le test). La borne haute sert de diamètre pessimiste pour l'étape 7.
- 41 % du catalogue est plus petit que 98 % des objets d'entraînement (H > 24,7) et porte le drapeau `hors_domaine`.

## Constats clés de la classification

- Cible : `pha`, prédit sans H ni MOID. Modèle retenu : Random Forest sur l'orbite et les variables d'observation, avec `class_weight="balanced_subsample"`. Le notebook prend environ 5 minutes à s'exécuter (recherche en grille).
- Le seuil de décision est choisi sur le train pour atteindre 99 % de rappel (seuil ≈ 0,055 au lieu de 0,5).
- Sur le test : 99 % des PHA attrapés (5 ratés sur 508) en surveillant 33 % du catalogue, contre 73 % avec la règle « l'orbite croise celle de la Terre » et 99 % au hasard.
- Avec l'orbite seule, il faut surveiller 72 % du catalogue pour le même rappel : l'essentiel du gain vient des variables d'observation, en partie parce qu'elles reflètent la taille.
- Les PHA ratés sont presque tous des objets découverts depuis quelques jours ou semaines : ils doivent être surveillés systématiquement, quel que soit leur score.

## Constats clés du clustering

- K-Means sur quatre variables standardisées : log du diamètre (étape 4), log du MOID, inclinaison et excentricité. `pha` ne sert qu'à décrire les groupes après coup.
- Choix des variables vérifié par corrélations et ACP : `a`, `q` et `ad` sont redondantes (r jusqu'à 0,98), et aucune des 4 variables retenues n'est superflue (3 composantes nécessaires pour dépasser 90 % de l'information). L'ACP sert à visualiser les clusters dans le plan CP1/CP2.
- K = 4 retenu (silhouette ≈ 0,27). Les groupes se chevauchent, car les géocroiseurs forment un continuum, mais ils sont reproductibles (ARI ≈ 0,99 entre relances).
- Les clusters ne redécouvrent pas les familles orbitales (ARI ≈ 0,08) : trois sur quatre mélangent Apollo et Amor à parts égales.
- Le cluster des gros objets proches de l'écliptique contient 64 % des PHA du catalogue : il justifie une campagne de suivi prioritaire.
- Le cluster des petits objets frôleurs (MOID médian 0,005 UA) contient 44 % d'objets passés ou attendus à moins de 5 distances lunaires, mais seulement 1 % de PHA : suivi réactif à la découverte.

## Constats clés de la politique de priorisation

- Trois niveaux : niveau 1 (observer ce cycle), niveau 2 (prochain cycle), niveau 3 (surveillance passive). Tous les seuils sont regroupés dans la première cellule du notebook.
- Niveau 1 = PHA probable (MOID ≤ 0,05 UA et diamètre estimé ≥ 140 m) **et** urgent (approche à moins de 20 distances lunaires d'ici 2035, orbite incertaine avec `condition_code` ≥ 7, ou découverte depuis moins d'un an) : 730 objets (1,8 % du catalogue).
- Niveau 2 = PHA probables non urgents, PHA possibles (seule la borne haute du diamètre dépasse 140 m) et découvertes récentes signalées par le modèle de l'étape 5 : 4 880 objets (11,8 %).
- 99,5 % des PHA officiels sont en niveau 1 ou 2. Les 12 autres ont un diamètre mesuré inférieur à 140 m.
- La politique ajoute 662 PHA probables et 2 184 PHA possibles absents de la liste officielle, car plus sombres que l'albédo de 0,14 supposé.
- Temps de télescope (hypothèses illustratives) : 81 % de moins que « tout surveiller », 74 % de moins que la règle d'expert, environ 50 % de plus que la liste officielle, mais réparti selon l'urgence.
- `file_priorite.csv` classe tout le catalogue avec les raisons de chaque décision : c'est la source de données prévue pour le Streamlit.
