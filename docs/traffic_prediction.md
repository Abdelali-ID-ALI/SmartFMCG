# Phase 5 — Prédiction des temps de trajet

## Données et séparation temporelle

L'expérience utilise le checkpoint `108d2fd` et ses 10 000 observations sur 930 arcs dirigés à Casablanca. L'alignement distance/durée avec l'instantané OSRM qualifié est vérifié avant entraînement. Les six CSV sont régénérés à l'identique. Le rapport durée simulée/durée statique respecte le plancher du générateur de 0,8 à la précision du CSV. Le maximum observé est 2,582.

| Ensemble | Dates en 2026 | Jours | Trajets |
| --- | --- | ---: | ---: |
| Entraînement | 6 avril–9 mai | 34 | 6 095 |
| Validation | 10–20 mai | 11 | 1 939 |
| Test | 21–31 mai | 11 | 1 966 |

Les données quotidiennes restent ensemble. Les modèles, catégories, normalisation linéaire et regroupements historiques sont ajustés sur les 34 premiers jours. La date sert à la séparation et à la dérivation du calendrier. Ni les identifiants bruts de trajets, ni la durée réalisée, ni le label d'incident contemporain n'entrent dans le modèle. Les zones sont issues de la table clients, et les cycles, l'heure, le week-end et la pointe sont calculés à partir du départ local. La relation intra/interzone est une catégorie synthétique. Les descripteurs de calendrier inspirés du générateur sont une hypothèse expérimentale documentée.

## Cibles et décision

Pour une base statique s et une durée simulée y, l'estimation directe apprend y. L'autre cible apprend y/s puis multiplie la prédiction par s. Une erreur de 0,1 dans cette seconde formulation vaut une minute si s = 10 et deux si s = 20. L'estimation des observations est toujours évaluée en minutes. La baseline historique utilise le ratio moyen par heure locale et type de jour, avec une moyenne globale d'entraînement pour les cases absentes. La régression utilise les moindres carrés. Random Forest agrège des arbres rééchantillonnés, XGBoost et LightGBM ajustent des arbres séquentiels. On teste deux réglages par famille d'arbres et cible. La MAE de validation choisit le réglage et le modèle, avec RMSE comme départage. Le modèle retenu reste ajusté sur l'entraînement seul.

## Résultats mesurés sur le test

| Modèle | Cible | MAE min | RMSE min | R² | Biais min |
| --- | --- | ---: | ---: | ---: | ---: |
| OSRM statique | — | 3,108 | 4,629 | 0,695 | −2,884 |
| Moyenne historique | ratio | 1,410 | 2,294 | 0,925 | +0,050 |
| Régression linéaire | minutes | 1,676 | 2,492 | 0,912 | −0,028 |
| Régression linéaire | ratio | **1,288** | **2,176** | **0,933** | −0,059 |
| Random Forest | minutes | 1,359 | 2,261 | 0,927 | −0,035 |
| Random Forest | ratio | 1,295 | 2,190 | 0,932 | −0,046 |
| XGBoost CPU | minutes | 1,353 | 2,224 | 0,930 | −0,054 |
| XGBoost CPU | ratio | 1,291 | 2,182 | 0,932 | −0,047 |
| LightGBM | minutes | 1,323 | 2,215 | 0,930 | −0,059 |
| LightGBM | ratio | 1,296 | 2,195 | 0,931 | −0,051 |

La sélection linéaire sur le ratio avait une MAE de validation de 1,259 min. Le test confirme un gain de 8,67 % en MAE sur l'historique, et un avantage descriptif très faible sur XGBoost ratio (0,003 min). Les quatre familles ont une meilleure MAE sur les journées de test avec la cible ratio. Cette comparaison concerne un découpage et le même mécanisme simulé. L'expérience complète a duré 17,50 s avec un pic RSS de 216 420 KiB. L'inférence sélectionnée a pris une médiane de 28,59 ms pour 1 966 trajets, calcul de variables compris, parmi cinq appels après échauffement. Le notebook avec SHAP a pris 7,15 s et 248 012 KiB de pic RSS au total.

## Vérification des raccourcis et limites

Avec la géométrie seule, le modèle linéaire sur le ratio atteint encore R² = 0,844 et MAE = 2,307 min. La base OSRM structure fortement la dispersion des durées. Supprimer les indicateurs explicites de pointe et de centralité dégrade la MAE à 2,109 min. La structure du simulateur explique la bonne adéquation de la cible ratio et des indicateurs disponibles. Les variables corrélées limitent l'interprétation causale de l'ablation et des attributions.

Les 75 incidents du test ont une MAE de 7,168 min et un biais de −7,168 min. Les 1 891 trajets sans incident ont une MAE de 1,055 min. L'intensité la plus forte a un biais de −1,532 min. La MAE augmente de 0,200 min jusqu'à 3 km à 1,986 min au-delà de 8 km. Les résidus signés et les détails par heure, intensité, distance et incident se trouvent dans les CSV d'évaluation. L'importance par permutation sur validation mesure la hausse de MAE, y compris l'effet d'échelle de la base OSRM. SHAP explique la sortie ratio avec un fond de 24 lignes d'entraînement et 40 lignes de test. Les contributions et la valeur de base sont vérifiées par additivité. Les variables liées et les valeurs hybrides de l'explication limitent son interprétation.

Une piste externe est le jeu public de Rouky et Bousouf sur 440 trajets Waze pendant une semaine à Casablanca. Un appariement géographique et temporel serait nécessaire pour en faire une validation externe. Les 11 jours testés ici sont simulés sur le même réseau et par le même générateur. **Cette performance n'établit pas la précision prédictive en circulation réelle.** [Jeu public de Rouky et Bousouf](https://data.mendeley.com/datasets/8hj76bpzpv/1)

## Artefacts

Le fichier `models/traffic_model.joblib` contient le modèle, son pipeline ajusté, la cartographie des zones et le mode cible. `predict(frame)` consomme les six colonnes définies dans `src/ml/features.py` et produit des minutes positives. Le rechargement a conservé à l'identique les prédictions du test. Les observations simulées ne sont jamais remplacées par des prévisions. Le [notebook exécuté](../notebooks/05_traffic_prediction.ipynb) et les [résultats](../results/experiments/traffic_ml) documentent l'expérience et les empreintes des données.
