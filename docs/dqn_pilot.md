# Phase 9A — Pilote scientifique DQN sur CPU

## Question testée

Deux tournées peuvent partager une case du Q-Learning alors que leurs créneaux ou leurs réserves rendent utiles des interventions différentes. La phase 8.1 a observé ce phénomène et une instabilité des choix entre checkpoints. À l’inverse, détailler des classes augmente le nombre de valeurs à estimer et réduit la densité des visites. Le pilote vérifie qu’un réseau peut recevoir les marges et stocks continus, produire quatre valeurs d’action sûres et être entraîné puis relu sur CPU. Il mesure aussi son coût et compare deux entrées sous la même architecture. [Diagnostic tabulaire](rl_refinement.md)

## Principe et exemple numérique

| Méthode | Ce qui est appris |
| --- | --- |
| Q-Learning tabulaire | Une valeur par paire état-action. La mise à jour touche la case visitée. |
| DQN | Un réseau de valeurs qui partage les paramètres entre observations et actions. Le réseau cible sert à évaluer le maximum admissible dans l’état suivant. |
| Double DQN | Le réseau courant choisit l’action suivante. Le réseau cible évalue ce choix. La séparation vise le biais de surestimation lié au maximum. |
| PPO | Une politique et une fonction de valeur sont ajustées sur des trajectoires récentes à l’aide d’un objectif de politique qui écrête la contribution du ratio. |

DQN mémorise `(observation, action, récompense, observation suivante, fin)` dans un tampon. Le rejeu échantillonne des transitions anciennes et récentes pour entraîner le réseau par lots. Une politique ε-gloutonne explore par un tirage uniforme parmi les actions admissibles et exploite sinon le maximum des valeurs. Pour une transition, la cible de différence temporelle vaut `y = r + γ (1 − d) max Q_cible(o′, a′)`, avec maximum sur les actions admissibles et `d = 1` à la terminaison réelle. Une limite externe de pas est enregistrée comme troncature et conserve le terme de continuation à partir de l’observation terminale de cette transition.

Exemple à la main. Pour `r = −2,5`, `γ = 0,9` et `max Q_cible = 4`, la cible vaut `1,1`. Si le réseau courant renvoie `0,6` pour l’action jouée, l’erreur vaut `0,5`. La perte de Huber à seuil 1 vaut `0,5 × 0,5² = 0,125`. Adam ajuste les poids par gradient et le réseau cible est resynchronisé périodiquement. Le maximum sur des estimations bruitées peut surestimer la valeur. Le couplage entre données produites par la politique, cibles estimées et approximation peut rendre l’apprentissage instable. L’observation des tournées reste partielle, car le trafic réalisé et les événements futurs restent cachés. Un réseau statique ne reconstitue pas automatiquement cette information. [DQN](https://doi.org/10.1038/nature14236) · [Double DQN](https://arxiv.org/abs/1509.06461) · [PPO](https://arxiv.org/abs/1707.06347)

## Contrat et comparaison contrôlée

PyTorch CPU 2.10.0 et Stable-Baselines3 2.8.0 réalisent les mises à jour. Le MLP a 22 entrées, deux couches de 64 neurones ReLU et quatre sorties. Le premier attribut est le bit public d’admissibilité des interventions. L’entrée continue conserve les 22 indicateurs bornés de phase 8.1. Le témoin encode les classes de phase 8 en indicatrices et complète le vecteur par des zéros pour maintenir la même dimension de réseau. Les deux régimes ont les mêmes capacités, les mêmes coûts, le même calendrier et les mêmes paramètres d’apprentissage. Les transformations sont déterministes et bornées, sans ajustement statistique sur les scènes.

Stable-Baselines3 DQN standard échantillonne dans l’espace discret pendant l’échauffement et l’exploration et maximise sur les sorties du réseau cible. Le pilote adapte ces trois points. Le bit public indique soit les quatre possibilités, soit le maintien seul. La sous-classe échantillonne parmi les actions disponibles pendant l’échauffement et l’exploration. Les deux réseaux excluent les actions indisponibles de leur maximum. Les sorties Q admises restent finies et le maintien garantit un maximum défini. L’éligibilité d’un appel de replanification n’anticipe pas le résultat du solveur. Le simulateur conserve ses contrôles indépendants de charge, temps, compatibilité et engagements. Un appel non admissible forcé dans les tests produit le maintien et le motif de repli. Les journaux distinguent demande, exécution, non-amélioration, statut solveur et repli. [Implémentation DQN de SB3](https://stable-baselines3.readthedocs.io/en/v2.8.0/_modules/stable_baselines3/dqn/dqn.html) · [Collecte de transitions de SB3](https://stable-baselines3.readthedocs.io/en/v2.8.0/_modules/stable_baselines3/common/off_policy_algorithm.html)

Le pilote retient la récompense originale en MAD et `γ = 1`, cohérent avec la perte non actualisée à horizon fini. Les intervalles irréguliers ne font donc varier aucun poids de retard via l’actualisation. La récompense anticipée est représentée dans le comparateur tabulaire D. Le pilote DQN change seulement l’information en entrée entre ses deux branches. Le tampon contient au plus 1 024 transitions, commence ses mises à jour après 96 interactions, utilise des lots de 32 et effectue un gradient toutes les quatre décisions. Le réseau cible est copié toutes les 128 décisions. L’exploration passe de 0,90 à 0,08 sur les premiers 80 % du budget. Les courbes mesurent la perte de Huber et les résultats opérationnels en validation.

## Protocole du pilote

Le développement utilise deux véhicules par jour. Les 12 scénarios d’entraînement et les six de validation occupent des dates distinctes. Dix autres scénarios constituent le contrôle indépendant du pilote. Ils sont générés après l’entraînement et leurs graines sont distinctes des artefacts des phases 8 et 8.1. Le test scientifique de 9B devra encore tirer de nouvelles scènes. La bibliothèque routière et les tournées de référence restent celles du projet. La variation porte sur la demande, le choix des véhicules, les réserves, la congestion et les urgences. Les règles physiques restent les mêmes.

Le pilote entraîne deux variantes DQN et trois variantes tabulaires, chacune sous deux graines, pendant 768 interactions par combinaison. A reprend les classes et la récompense de phase 8. B utilise les classes raffinées et la récompense originale. D ajoute la rétroaction anticipée. Le budget interrompt éventuellement la dernière journée d’apprentissage. Les mesures par épisode et toutes les évaluations opérationnelles n’utilisent que des journées arrivées à leur terme. Les six règles classiques sont celles déjà validées. Les tables de phase 8 servent aussi à un rejeu historique distinct, avec leur budget d’apprentissage d’origine. Les comparaisons apparient les mêmes cas et rééchantillonnent les dates après moyenne des graines. Les cinq dates et deux graines suffisent à tester le pipeline et à observer des ordres de grandeur.

Les checkpoints intermédiaires enregistrent le modèle avant la mise à jour qui suit la collecte du pas 384. Les checkpoints finaux incluent toutes les mises à jour. Les tampons finaux, configurations, transitions et mesures sont sauvegardés. Le chargement CPU vérifie l’identité des poids, valeurs et actions et recharge le tampon. Les journées simulées entières restent l’unité de comparaison. La reprise au milieu d’une journée exigerait en plus la sauvegarde du simulateur et de ses générateurs aléatoires.

## Résultats techniques et opérationnels

Les quatre réseaux ont chacun 5 892 paramètres et ont effectué 168 mises à jour. Chacun a produit 15 journées complètes et un tampon final de 768 transitions. Le changement de norme L2 des poids est compris entre 2,27 et 2,40. Les tests contrôlent le traitement des troncatures par le rejeu et la distinction avec une vraie fin dans la cible de Bellman. Les deux checkers passent. Les quatre modèles et leurs tampons se rechargent sur CPU. Huit relectures de contrôle reproduisent la perte, l’OTIF, les urgences, les tentatives et les appels solveur. Profils par modèle (historical artifact: `results/experiments/dqn_pilot/training_profile.csv`)

| Politique | OTIF total | OTIF régulier | Urgences servies | Exploitation (MAD) | Avec omissions (MAD) | Retard (min) | Tentatives | Replans appliqués | Changements | Appels CP-SAT | Perte (MAD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Statique | 80,52 % | 89,92 % | 0,00 % | 827,80 | 939,80 | 7,31 | 0,00 | 0,00 | 0,00 | 0,00 | 952,95 |
| À chaque événement | 82,60 % | 91,59 % | 10,00 % | 832,93 | 928,93 | 5,71 | 3,40 | 0,50 | 1,40 | 0,00 | 943,71 |
| Périodique | 83,37 % | 93,09 % | 0,00 % | 826,39 | 938,39 | 4,04 | 26,20 | 0,40 | 1,30 | 0,00 | 961,35 |
| À seuil | 83,37 % | 93,09 % | 0,00 % | 826,20 | 938,20 | 4,02 | 1,10 | 0,40 | 1,30 | 0,00 | 948,58 |
| Multicritère | 83,37 % | 92,42 % | 10,00 % | 830,93 | 926,93 | 4,87 | 7,10 | 0,80 | 1,80 | 0,00 | 942,84 |
| Aléatoire | 83,37 % | 93,09 % | 5,00 % | 826,82 | 930,82 | 4,22 | 22,60 | 0,60 | 1,40 | 6,90 | 952,51 |
| Table historique phase 8 | 83,11 % | 92,59 % | 3,33 % | 828,62 | 935,28 | 4,59 | 28,07 | 0,47 | 1,33 | 7,30 | 960,25 |
| Tabulaire A | 82,62 % | 92,26 % | 2,50 % | 827,41 | 935,41 | 5,00 | 15,25 | 0,40 | 0,95 | 9,90 | 953,94 |
| Tabulaire B | 83,37 % | 93,09 % | 0,00 % | 826,94 | 938,94 | 4,02 | 24,95 | 0,50 | 1,30 | 10,60 | 961,24 |
| Tabulaire D | 82,98 % | 92,67 % | 0,00 % | 827,59 | 939,59 | 4,75 | 24,55 | 0,40 | 1,05 | 10,40 | 962,51 |
| DQN compact | 80,88 % | 90,34 % | 0,00 % | 827,86 | 939,86 | 7,15 | 1,30 | 0,05 | 0,15 | 0,00 | 953,67 |
| DQN continu | 80,52 % | 89,92 % | 0,00 % | 827,80 | 939,80 | 7,31 | 0,00 | 0,00 | 0,00 | 0,00 | 952,95 |

Les taux sont moyennés par scénario, puis par graine pour les agents entraînés. Les deux réseaux continus sélectionnent le maintien sur toutes les décisions du contrôle. Le DQN compact applique un replan au total sur vingt journées. L’écart de perte du continu face au compact est de −0,72 MAD, avec intervalle bootstrap à 95 % de −1,23 à −0,23. Cet écart provient principalement de tentatives évitées. Le continu perd 0,36 point d’OTIF et les deux variantes omettent toutes les urgences. Face au multicritère, le continu perd 2,84 points d’OTIF et coûte +10,12 MAD, avec intervalle de −3,23 à +28,60 MAD pour la perte. L’écart de perte face à la table A vaut −0,98 MAD avec intervalle de −7,30 à +6,53. Aucun appel illégal ni repli de sécurité n’a été relevé sur le contrôle. Les journaux séparent ces indicateurs de l’absence de plan améliorant. Comparaisons appariées (historical artifact: `results/experiments/dqn_pilot/paired_intervals.csv`) · Décisions (historical artifact: `results/experiments/dqn_pilot/pilot_decisions.csv`)

En validation, la perte moyenne du continu passe de 1 024,76 MAD à l’initialisation à 1 009,51 après 384 interactions, puis à 1 025,97 après 768. Les tentatives passent de 15,58 à zéro entre les deux checkpoints. Ce retour vers le maintien, le petit nombre de journées et la variance selon la graine limitent l’interprétation. Les valeurs Q changent tout au long du pilote. Le comportement final ne montre aucune amélioration du service urgent. Le test de 9B devra distinguer l’effet du budget, de la représentation et du signal terminal sans réutiliser ces dix cas pour conclure. Courbes de validation (historical artifact: `results/experiments/dqn_pilot/validation_curves.csv`)

## Ressources et proposition pour 9B

| Mesure observée | Valeur |
| --- | ---: |
| Pas de simulateur moyen dans l’entraînement DQN | 5,07 ms |
| Pas de simulateur, 95e percentile | 35,21 ms |
| Pas avec demande de réoptimisation locale | 31,57 ms |
| Journée de simulation, réinitialisation incluse | 0,264 s |
| Appels CP-SAT par journée d’entraînement DQN | 4,73 |
| Échantillonnage et mise à jour du réseau | 3,84 ms |
| Inférence déterministe par décision | 0,410 ms |
| Temps complet du pilote | 117,55 s |
| Pic RSS du pilote | 604 464 KiB |
| Notebook exécuté | 7,50 s et 156 292 KiB |
| 109 tests usuels | 26,49 s et 554 304 KiB |
| 35 régressions d’audit | 5,12 s et 147 628 KiB |

La planification représente **86,5 %** du temps des pas de simulation observés. L’acquisition et l’apprentissage prennent 6,85 ms par interaction DQN en moyenne, hors évaluations de validation, et 6,23 ms par interaction tabulaire. Les durées varient selon le mélange d’actions et la taille des suffixes. Le solveur CPU et le nombre de ses appels dimensionnent la montée en charge. La version CPU de PyTorch et la prévision locale suffisent pour le budget proposé. Le notebook, les suites d’audit, le checker Gymnasium, celui de Stable-Baselines3 et `pip check` ont été exécutés avec succès. [Dépendances phase 9A](../requirements-dqn.txt) · Notebook (historical artifact: `notebooks/09_dqn_pilot.ipynb`)

**Proposition soumise à approbation.** Commencer par deux taux d’apprentissage, `3e−4` et `1e−3`, à 2 000 interactions chacun sur une graine. Fixer architecture, récompense, masque et banques. Examiner sur validation la perte opérationnelle, les actions et les appels solveur. Si les deux réglages restent au maintien systématique, revoir le budget et le protocole à ce point avant d’étendre l’expérience. Pour la comparaison complète, plafonner à 4 000 interactions par combinaison, trois graines et deux entrées DQN. Le plafond est de **28 000 interactions DQN**, présélection comprise, plus **36 000 interactions tabulaires** pour A, B et D. Proposer une banque d’entraînement à trois véhicules et un test nouveau à trois et quatre véhicules, avec comparateurs et mêmes cas.

Les débits du pilote donnent environ sept minutes de calcul pour les interactions seules. Prévoir **10 à 16 minutes sur CPU** pour les cas plus grands, la validation et les comparaisons finales, avec un plafond proposé de **20 minutes** et une revue si le profil initial l’indique. Le budget reste une proposition. La phase 9B attend une approbation explicite.

## Reproduction

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-dqn.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
/usr/bin/time -v python -m src.rl.dqn_pilot
jupyter execute notebooks/09_dqn_pilot.ipynb --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
```

[Configuration](../configs/dqn_pilot.yaml) · Artefacts complets et checkpoints CPU (historical artifact: `results/experiments/dqn_pilot`)
