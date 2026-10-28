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

## Constats clés de l'EDA

- `pha` est égal à la règle MOID ≤ 0,05 UA et H ≤ 22 pour 99,9 % des objets. Il faut exclure MOID, H et les colonnes dérivées de H si `pha` est la cible.
- 97 % des diamètres sont calculés à partir de H avec un albédo fixe de 0,14. Seuls 1 246 diamètres sont réellement mesurés.
- La chute des approches après mars 2026 vient de la date d'extraction : les objets pas encore découverts n'ont pas de prévisions.
- 80 % des objets n'ont qu'une seule approche dans la fenêtre 2015–2035.
