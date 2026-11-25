# Phase 6 — Predict-Then-Optimize

## Formulation et choix

Le problème reste le HF-VRPSPD-TW validé. Pour le véhicule v et un arc dirigé i→j choisi, l'heure du service suivant est au moins l'heure du service courant, plus le service et le temps de trajet. La phase 6 remplace uniquement ce dernier terme. La distance, la flotte, les créneaux, les charges, les coûts unitaires et les règles métier restent ceux de l'instance de référence. Le validateur rejoue chaque plan avec la matrice qui l'a produit.

Le pipeline linéaire du checkpoint `d0d3100` estime un ratio sur la durée OSRM. Par exemple, un arc de 10 min avec un ratio 1,3 devient 13 min, soit 780 s pour le solveur. La prédiction reçoit le calendrier, les zones et la géographie statique. Le remplissage d'une matrice arrondit les durées vers le haut à la seconde et préserve la diagonale, les arcs indisponibles et les éventuels accès communs qualifiés. Les prévisions doivent être finies et strictement positives sur les arcs de durée positive.

- **OSRM statique.** Conserver la base qualifiée dirigée.
- **Historique.** Appliquer à OSRM le ratio historique moyen par heure et type de jour, calculé sur l'entraînement seul. La matrice est figée à 11 h 30.
- **ML fixe.** Appliquer le ratio linéaire par arc pour 11 h 30. L'heure commune est un réglage expérimental choisi à partir des créneaux, utilisable avant le départ des véhicules. Elle sert de point d'ancrage aux trois variantes prédictives.
- **ML itératif.** Commencer avec la matrice ML fixe, reclasser les arcs parcourus selon l'heure de départ indiquée par le dernier plan et recalculer ces arcs au milieu de leur tranche horaire. Les autres reviennent à la référence. Arrêter à convergence ou à trois appels sur le profil standard. Conserver le dernier plan valide si une itération échoue. L'horaire d'un arc peut encore changer après le dernier appel. Chaque plan est valide pour sa matrice figée, avec une approximation de la dépendance horaire.
- **ML avec marge.** Ajouter au ratio ML 0,0887, le quantile 80 % des résidus normalisés, définis par durée réalisée moins durée prédite. Le quantile est calculé sur la distribution complète des résidus signés de validation et la marge est bornée inférieurement à zéro. Elle ajoute environ 53 s à un arc OSRM de 10 min. Ce calibrage marginal ne garantit ni la couverture des incidents rares ni celle d'une tournée entière.

Le solveur reçoit jusqu'à 0,9 s de budget total par méthode et journée. L'itération partage ce budget en trois appels, ce qui change l'effort disponible pour chaque matrice. L'expérience ne mesure pas séparément cet effet et celui des mises à jour horaires. L'inférence pré-calcule par lot les 24 heures une fois par jour sur CPU. Les empreintes de la matrice et du trafic sont contrôlées avec le manifeste de phase 5.

## Évaluation appariée

Les 11 dates de test vont du 21 au 31 mai 2026. Pour chaque date et arc-heure, un nouveau tirage synthétique applique le mécanisme de la phase 5 avec une graine d'évaluation commune aux stratégies. Le choc d'incident a une probabilité de 3,5 % et ajoute 0,65 au ratio. Le bruit normal et l'incident sont tirés par arc-heure et persistent dans cette case. Les réalisations n'entrent dans aucun plan et les dates de test n'entrent dans aucune calibration. La même flotte et les mêmes commandes sont utilisées chaque jour.

Le rejeu maintient les départs du dépôt et la séquence, puis applique à chaque jambe sa durée réalisée selon l'heure courante. Les services commencent après l'arrivée et l'ouverture. L'OTIF compte les commandes livrées intégralement et dont le service finit au plus tard à la fermeture, sur les 30 commandes. Le retard est la somme des dépassements de fermeture, en minutes. La faisabilité réalisée exige zéro retard, zéro dépassement au retour et la couverture des obligatoires. Elle est plus exigeante qu'un bon taux moyen. Pour mesurer l'ampleur des écarts, les services tardifs sont poursuivis dans le rejeu. Le coût réalisé additionne le fixe, la distance, le temps écoulé dont l'attente et les omissions éventuelles. Les retards eux-mêmes sont rapportés séparément. La distance, les quantités servies et les charges restent fixées par le plan, sans annulation ni replanification. L'utilisation ci-dessous est la moyenne du poids initial rapporté à la capacité de chaque véhicule actif.

## Mesures du profil standard

| Stratégie | Coût planifié MAD | Coût réalisé MAD | OTIF réalisé | Retard min/j | km/j | Util. poids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OSRM statique | 2 088,07 | 2 113,50 | 87,27 % | 13,15 | 116,61 | 84,44 % |
| Historique | 2 105,29 | 2 113,61 | 90,61 % | 8,89 | 113,23 | 84,57 % |
| ML fixe | 2 104,02 | 2 111,76 | 89,39 % | 10,13 | 120,36 | 85,55 % |
| ML itératif | 2 132,85 | 2 140,62 | 90,00 % | 9,90 | 130,00 | 85,62 % |
| ML avec marge | 2 122,49 | 2 115,41 | **94,85 %** | **3,75** | 121,14 | 85,27 % |

Les 55 plans passent la validation et servent les 30 commandes avec cinq véhicules. L'OTIF planifié est de 100 %. Après rejeu, chacun comporte au moins un retard, et aucun retour ne dépasse les horaires. La marge améliore l'OTIF sur les 11 jours par rapport à OSRM. Son écart moyen de coût est de +1,91 MAD et varie selon le jour de −23,31 à +23,85 MAD. L'itération coûte en moyenne +27,13 MAD par rapport à OSRM et coûte davantage les 11 jours. Le ML fixe présente un coût moyen de −1,73 MAD face à OSRM et est moins cher six jours sur onze. Ces variations et le nombre limité de jours invitent à traiter les rangs de coût comme descriptifs.

Sur la grille indépendante des 930 arcs et des heures 7 h à 17 h, la MAE horaire moyenne du ML est de 1,265 min contre 1,394 pour l'historique. Les variantes à matrice figée ont des erreurs de 2,434 et 2,493 min, contre 3,301 pour OSRM. Malgré ces écarts, l'historique fixe est plus ponctuel que le ML fixe. En condition d'incident, le ML horaire atteint 6,811 min de MAE. Les moyennes de prévision ne pondèrent pas les créneaux et affectations sensibles d'un plan.

L'expérience standard a duré 55,81 s avec un pic RSS de 218 988 KiB. La planification prend en moyenne 0,906 à 0,933 s par stratégie et jour. La préparation horaire partagée prend en moyenne 0,115 s par jour et le rejeu environ 0,0004 s par plan. Le notebook a été exécuté dans IPython avec Matplotlib Agg en 2,66 s et 133 292 KiB de pic RSS. Le solveur domine le temps observé, et les incumbents peuvent changer avec le budget mural. Les autres profils sont configurés pour un usage local progressif. Aucun entraînement, essai GPU ou calcul cloud n'a été effectué pour cette phase.

## Reproduction

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -m src.planning.experiment --profile standard
jupyter execute notebooks/06_predict_then_optimize.ipynb --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
```

Le profil `development` réduit le nombre de jours et le temps de recherche. Les profils non standard écrivent par défaut dans des dossiers séparés. Le paramètre `--output` permet de choisir la destination. Les plans, prévisions agrégées et indicateurs appariés sont dans [results/experiments/predict_then_optimize](../results/experiments/predict_then_optimize). Les résultats restent propres au générateur, au réseau et à la seule demande de référence. Une validation externe, l'adaptation en temps réel et l'évaluation à plusieurs tailles relèvent de travaux ultérieurs.

**GO pour la phase 7**, sous réserve d’approbation. Les retards réalisés et les trajectoires validées fournissent un cas de référence pour étudier l’adaptation en cours de journée.
