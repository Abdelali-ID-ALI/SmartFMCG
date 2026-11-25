# Guide du tableau de bord SmartFMCG

## Lancer la démonstration

Se placer à la racine du dépôt. L'environnement testé utilise Python 3.14.2 et PyTorch CPU sur Linux x86‑64.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: pip==26.2
python -m pip install --only-binary=:all: -r dashboard/requirements.txt
python -m pip check
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -m streamlit run dashboard/app.py
```

Le tableau de bord local s’ouvre sur les résultats archivés à l’adresse `http://localhost:8501`. L’environnement de démonstration utilise Streamlit 1.67.0, Pillow 13.0.0 et la roue PyTorch 2.13.0+cpu. Les contraintes de `requirements-lock.txt`, `requirements-dqn.txt` et `dashboard/requirements-lock.txt` sont appliquées par le manifeste de démonstration. Installer aussi `requirements.txt` selon le guide local pour exécuter les notebooks, pytest et les expériences de comparaison de modèles. [Installation et versions](local_installation.md)

## Comparer un plan fixe et une politique adaptative

1. Dans **Scenario configuration**, conserver le scénario léger de 12 clients et la graine 317, ou choisir une taille, les modèles de véhicules et le niveau de perturbation. **Build scenario** vérifie la cohérence et affiche les commandes et le parc dimensionné. Le bouton est facultatif pour les paramètres par défaut. L'interface limite le parc à 16 véhicules disponibles et signale quand le dimensionnement nécessite plus de véhicules ou un frigorifique.
2. Dans **Route planning**, sélectionner un ou plusieurs des quatre modèles de temps de trajet et choisir **Generate plans**. Comparer coûts prévus et statuts du solveur. Choisir un plan pour voir son trajet, ses arrivées, le service planifié et les charges. Les volets affichent les quatre dimensions de capacité.
3. Dans **Dispatch replay**, choisir B5 par défaut, B8 pour la règle multicritère, ou l'un des comparateurs. **Start paired simulation** exécute la journée pour la politique choisie et pour B4 sur le même plan robuste. **Advance**, le curseur et **Replay from start** parcourent les états produits aux occasions de décision. **Reset replay** ferme la lecture. Les résultats déjà calculés restent réutilisables dans la session.
4. À la fin du rejeu, **Realized day comparison** affiche la comparaison. **Overview** garde le dernier résultat actif et donne accès au benchmark archivé. La perte inclut coût d'exploitation, omissions, retards, dépassements d'horaires, modifications et tentatives. Une économie positive correspond à la perte B4 moins la perte de la politique. L'OTIF rapporte les commandes terminées à temps à l'ensemble des commandes connues en fin de journée. Le service urgent inclut les commandes urgentes honorées en retard.
5. **Travel-time prediction**, **Learning policies** et **Scientific benchmark** présentent les protocoles et mesures déjà produits. Les graphes affichent les unités, la taille des échantillons et l'incertitude disponible. Les comparaisons entre politiques suivent leur protocole apparié.

Une flèche schématique relie des points du simulateur sur la carte. L'état d'un véhicule en transit donne son dernier nœud et sa destination. Le moteur révèle les événements selon leur chronologie. L'heure opérationnelle s'écoule dans le simulateur et les temps muraux mesurent le calcul. À taille élevée, prévoir une attente lors de la préparation de la matrice horaire et des deux rejeux. La recherche initiale dispose de 0,65 s jusqu'à 50 clients et de 1,80 s à 100 clients. Le dispatch limite l'examen à 400 candidats pour les actions concernées et utilise le budget CP-SAT archivé. Les plans bornés en temps peuvent varier selon la machine et la charge.

## Données et mode hors ligne

| Élément | Artefacts versionnés |
| --- | --- |
| Géographie routière OpenStreetMap via OSRM | `data/matrices/` et `data/scalability/`, fichiers qualifiés par les manifestes et les configurations. Le seul tronçon archivé avec une géométrie routière détaillée relie D01 à C30 dans ce sens. |
| Demande, véhicules et apprentissage historique | `data/synthetic/`, `configs/`, `models/traffic_model.joblib`, manifestes et plans des phases 5 et 6. |
| Politiques apprises gelées | `results/experiments/dqn_evaluation/frozen_protocol.json` et les fichiers de modèles désignés dans `fit/`. Les modèles compacts suivent `results/experiments/final_benchmark/approximation_protocol.json`. Les SHA-256 sont vérifiés au chargement. |
| Références d'analyse | CSV de `traffic_ml/`, `predict_then_optimize/`, `dqn_evaluation/`, `final_benchmark/` et `scalability/` dans `results/experiments/`. Les graphiques SHAP sont lus depuis les sorties de `notebooks/05_traffic_prediction.ipynb`. |

La prévision, les plans et le rejeu fonctionnent avec les artefacts locaux après installation. La carte Folium charge des scripts Leaflet et des tuiles publiques dans le navigateur. Activer **Map without external tiles or scripts** pour une carte interactive en coordonnées, sans fond de carte distant. Les traits pleins représentent une géométrie OSRM archivée quand le tronçon et le sens correspondent. Les traits pointillés sont des liens schématiques. Les distances et durées viennent des tables dirigées. Les temps de base OSRM sont des valeurs statiques. Les observations de trafic sont simulées.

Le service d’inférence attend les artefacts du checkout qualifié et leurs empreintes. Le prédicteur joblib, les Q-tables et les ZIP DQN sont lus depuis leurs emplacements de projet. Conserver les modèles et manifestes de référence en lecture seule dans la copie locale. Les tampons RL servent aux expériences historiques. [Attribution des données](../data/ATTRIBUTION.md) · [Licences et droits des données](../DATA_LICENSE.md)

## Exécution depuis GitHub

Le dépôt proposé réunit le code et la documentation. Les analyses et scénarios exigent également les artefacts qualifiés. Suivre la procédure d’acquisition autorisée et de vérification SHA-256 avant le lancement. La disponibilité du paquet complet conditionne la reproduction des sept vues. Ouvrir la vue **Overview**, construire le scénario par défaut, activer la carte sans tuiles dans **Route planning** et effectuer un rejeu B4–B5 pour vérifier l’installation. GitHub sert à distribuer le projet. L’application s’exécute localement. [Installation et intégrité des artefacts](local_installation.md)

## Présentation technique et entretien

Pour une présentation courte, partir du retour d'emballages et expliquer la charge intermédiaire avec l'exemple du notebook. Démontrer ensuite le modèle de ratio et la marge, afficher les tournées validées et rejouer le même scénario avec B4 et B5. Conclure par le compromis perte–OTIF du benchmark apparié, les limites de transfert du DQN et le périmètre des données synthétiques. [Notebook exécuté](../notebooks/11_dashboard.ipynb)

**Formulation CV** — Conception d'un démonstrateur d'aide à la décision pour la distribution FMCG à Casablanca, couplant prévision des durées, optimisation de tournées sous contraintes et replanification par simulation événementielle. Validation sur des données synthétiques avec réseau OSRM archivé, jusqu'à 100 clients et 16 véhicules disponibles.

## Captures de la démonstration locale

La galerie présente les sept vues de la démonstration exécutée. [Galerie](dashboard_gallery.md)
