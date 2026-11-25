# Phase 8 — Décisions adaptatives et Q-Learning tabulaire

## Pourquoi apprendre une politique de dispatch

Le problème métier porte sur le choix de conserver ou de modifier une tournée déjà engagée. Une replanification peut sauver un service urgent ou réduire un retard. Elle peut aussi décaler les prochains départs, mobiliser du calcul et perturber une tournée qui restait acceptable. La phase 8 dresse le bilan de ce compromis par journée. La construction d’une tournée et la vérification des contraintes restent du ressort du moteur de dispatch et de ses validateurs.

## Audit ciblé avant entraînement

Le checkpoint de départ est `f183ede`. L’audit de pré-entraînement a exécuté les 17 tests de la phase 7, dont la conservation dans les quatre dimensions, le déroulement des trajets et services engagés, le rejeu statique, le rejet d’une proposition devenue obsolète, le temps d’attente induit et les observations identiques avant la révélation de deux scénarios différents. Les journaux de mouvement sont réconciliés par le validateur indépendant à la fin de chaque journée. Les tests de phase 8 ajoutent la détermination par graine, l’action de secours, le contrôle du checker Gymnasium et la réconciliation de la somme des récompenses avec le coût terminal. Aucun changement des contraintes physiques n’a été nécessaire.

Le diagnostic des 33 urgences extrêmes reprend les stocks et événements versionnés. Il vérifie le froid, les restrictions d’accès et le stock disponible par véhicule et par produit. Puis, pour les demandes qui ont une réserve compatible, il rejoue la trajectoire statique jusqu’aux décisions éligibles. À chaque décision, il énumère les insertions dans l’ordre courant et, si nécessaire, toutes les permutations du suffixe du véhicule. Le validateur évalue les fenêtres à partir des annonces disponibles et de la latence de préparation. Le diagnostic d’infaisabilité par le stock vaut pour le modèle, où une commande doit être servie par une seule tournée et où le stock ne se transfère pas. Le diagnostic par permutation est conditionnel aux trajectoires statiques visitées, aux engagements déjà pris et aux prévisions.

Sur 33 urgences, 28 dépassent la réserve de tout véhicule compatible. Les cinq autres ont au moins une insertion compatible avec les contraintes physiques si l’on isole le contrôle des fenêtres à des fins de diagnostic. Dans deux cas, le service urgent direct est possible sous prévision, mais aucune permutation du suffixe ne satisfait toutes les fenêtres aux décisions étudiées. Les trois derniers disposent d’un témoin validé qui livre effectivement à temps en rejouant les conditions réelles. Deux témoins ont une fenêtre égale aux trois minutes de service. Le troisième nécessite un réordonnancement. La recherche globale de phase 7 ne sélectionne l’urgence dans aucun des trois états témoins initiaux. Pour les deux insertions directes, son score prévisionnel augmente d’au moins 22,08 MAD et 4,08 MAD respectivement, après déduction de la pénalité évitée. Les trois scénarios restent donc des échecs des politiques évaluées, avec une solution physiquement réalisable dans le modèle. Vérification des 33 cas (historical artifact: `results/experiments/q_learning/extreme_audit.csv`)

## Formulation séquentielle

Un MDP représente les décisions par un état, des actions, une loi de transition et une récompense. Un état complet ici comprend l’heure, les positions et engagements, les commandes et charges, la partie encore cachée du scénario et le calcul en cours. En connaissant cet état, l’évolution après une action dépend de l’état courant et des mécanismes du simulateur. En exploitation, un décideur voit les données publiques et une prévision. Les réalisations du trafic en cours et les annonces futures restent cachées. Deux états peuvent produire la même observation avec des marges, positions et stocks différents. Le problème observé est donc partiellement observable. L’agent tabulaire apprend sur cette agrégation et ne bénéficie pas des garanties de convergence du Q-Learning appliqué à un MDP tabulaire entièrement observé et stationnaire.

Une transition va d’une décision à la suivante ou à la fin de la distribution. Les annonces et les points réguliers toutes les 15 min sont partagés entre politiques. L’intervalle entre décisions peut varier. À la fin du jour, `terminated` indique que le journal est clos et les omissions sont réglées. Une limite externe de pas donne `truncated`, fournit l’observation suivante et laisse la continuation dans la cible de Bellman. En comparaison, toutes les journées vont à leur terme. [API Gymnasium](https://gymnasium.farama.org/v1.0.0/api/env/)

La mise à jour utilise la récompense `r`, le pas `α`, la valeur existante et la meilleure action disponible au prochain état.

```text
cible = r + γ^(Δt / 900) × max Q(observation_suivante, action)
Q ← Q + α × (cible − Q)
```

En fin naturelle de journée, le second terme de la cible est nul. À pas de durée identique, un facteur constant par décision suffit. Ici, un exposant proportionnel au temps traite deux intervalles de 450 s comme un intervalle de 900 s. `γ = 1` optimise la somme journalière non actualisée. `γ = 0,995` introduit une préférence pour les coûts évités plus tôt. Les deux variantes sont départagées sur validation. Exemple pour `Δt = 900`, `Q = 4`, `r = −2`, `max Q′ = 6`, `γ = 0,9` et `α = 0,5`. La cible vaut `−2 + 0,9 × 6 = 3,4` et `Q` devient `4 + 0,5 × (3,4 − 4) = 3,7`. Avec un état terminal, la cible est `−2` et la valeur devient `1`.

## Observation, actions et sécurité

Les quatre composantes discrètes sont le contexte (indisponible, ordinaire, perturbation, urgence avec réserve compatible, urgence sans réserve compatible), le risque de retard (nul, modéré, élevé), la charge restante (plus d’un tiers des commandes ou moins) et le temps depuis la dernière tentative (moins de 10 min ou autre). L’espace déclaré comporte 60 combinaisons. Le contexte indisponible utilise un état canonique, d’où au plus 49 combinaisons distinctes avant de tenir compte des dépendances. Les diagnostics publics incluent aussi l’heure, le nombre de véhicules prêts ou engagés, les stocks, la capacité libre sur les quatre dimensions, la congestion annoncée, la marge urgente estimée sur le trajet direct et les changements précédents. La marge directe est un indicateur, car elle peut ignorer d’autres fenêtres et le coût d’une modification. Le risque prévu et le stock compatible servent d’indicateurs de bénéfice potentiel. Le calcul de l’observation n’exécute aucun solveur. Les seuils sont fixés à partir des règles métier et des durées du simulateur. Ils ne sont pas ajustés sur les scénarios finaux.

| Action | Portée et calcul |
| --- | --- |
| `KEEP_PLAN` | Poursuit l’exécution. |
| `LOCAL_HEURISTIC` | Applique l’insertion et un déplacement sur une tournée selon la règle locale de phase 7. |
| `LOCAL_REOPTIMIZATION` | CP-SAT évalue les permutations du suffixe d’un véhicule sélectionné et l’insertion éventuelle d’une urgence compatible. |
| `GLOBAL_REOPTIMIZATION` | La recherche bornée de phase 7 examine les suffixes de la flotte et choisit le véhicule d’une nouvelle urgence parmi les réserves compatibles. |

Le mode CP-SAT encode les arcs par un circuit avec départ courant et dépôt final, les temps de fin par visite et les charges sur chaque dimension. Il conserve les visites planifiées et contrôle que le stock disponible peut alimenter l’urgence optionnelle. Les durées de trajet sont figées à l’ancre de calcul pour obtenir une approximation prudente des alertes déjà annoncées. Pour une solution retenue, le validateur vérifie ensuite le candidat avec la prévision dépendante de l’heure à la préparation et à la diffusion. La recherche est bornée en temps déterministe CP-SAT à un thread. Les cas sans amélioration, incidents de résolution, rejets et indisponibilités sont journalisés séparément. La recherche globale choisit parmi plusieurs véhicules selon l’heuristique. Elle n’est pas une preuve d’optimalité sur la flotte.

Le masque autorise l’action de maintien partout et les trois formes de calcul aux points où un véhicule est encore modifiable et où la diffusion est possible. Une action indisponible aboutit au maintien et expose sa raison. Le masque décrit la possibilité de lancer le calcul. Les identifiants de scénario sont réservés à la traçabilité expérimentale. Le test de faisabilité du candidat intervient ensuite. Les traces donnent l’action demandée et celle appliquée, les statuts, les appels solveur, le temps mural et la latence simulée. Les mesures murales sont disponibles hors de l’observation et de la récompense pour préserver la détermination des transitions.

## Récompense

La récompense entre deux décisions est l’opposé de l’augmentation du coût cumulé en MAD. Le coût d’exploitation inclut l’activation au départ, la distance à l’arrivée et le temps écoulé à bord, y compris l’attente de diffusion en tournée. On ajoute 1,80 MAD par minute de retard livré, 3 MAD par minute de dépassement horaire au retour, 2 MAD par insertion ou changement de prédécesseur et 0,50 MAD par tentative effectivement lancée. Ces 0,50 MAD représentent un coût de coordination synthétique. Vingt secondes de latence représentent un délai de diffusion. Le temps mural du solveur ne figure pas dans la perte, afin de préserver la reproductibilité entre postes. Le temps d’attente physique figure une seule fois dans le coût d’exploitation.

Une urgence servie évite son coût d’omission. Une urgence non servie coûte 80 MAD, et une commande obligatoire omise, si le moteur en produit une, coûte 250 MAD à titre diagnostique. Les omissions sont liquidées une seule fois en fin de journée. Les contraintes de stock et de compatibilité sont contrôlées avant application indépendamment de ces coûts. Ignorer une urgence peut rester rationnel si son service dégrade davantage le reste de la journée. Cet arbitrage se lit dans l’OTIF régulier et urgent séparé. La coordination, l’instabilité et l’attente découragent les replans répétitifs. Une troncature ne donne ni récompense de fin ni sortie artificielle dans les expériences. Le masque écarte les tentatives structurellement invalides et les journaux rendent les échecs visibles. La sensibilité fait varier un facteur à la fois, avec 40, 80 et 120 MAD d’omission ou 5, 20 et 60 s de diffusion, exclusivement sur les jours de validation.

## Protocole

Les scénarios utilisent des sous-ensembles de tournées complètes du plan ML avec marge, deux véhicules en développement et trois en standard. Le nombre d’arrêts, la composition de la flotte et les contraintes d’accès varient avec les tournées sélectionnées. Une à deux caisses peuvent être retirées d’une commande régulière en conservant les reprises, et le préchargement de réserve propose 4, 8 ou 12 caisses selon le scénario avec plafonnement physique. Les fenêtres, vitesses, compatibilités et capacités restent celles du modèle validé. Les scénarios synthétiques tirent des horaires et des produits urgents selon trois régimes. Les dates et espaces de graines sont disjoints entre entraînement, validation et test. Le pipeline ML et les plans de la phase 6 sont utilisés comme références figées.

La table éparse initialise les valeurs à zéro. L’exploration tire une action disponible avec une probabilité `ε` décroissante de 0,90 à 0,08. L’exploitation choisit l’action disponible de meilleure valeur. À l’évaluation, une observation inconnue donne le maintien et les actions sans visite sont écartées parmi les choix d’un état connu. Les graines d’apprentissage changent l’ordre des journées et les explorations. Les checkpoints JSON conservent les valeurs, visites, paramètres, graines et épisodes. Le dernier checkpoint de chaque graine est rechargé avant inférence finale. Les six comparateurs reçoivent les mêmes scénarios et horloges. Le comparateur aléatoire tire uniformément parmi les options structurellement disponibles.

L’entraînement standard sélectionne les hyperparamètres sur validation après deux essais de 80 épisodes, puis entraîne trois tables de 180 épisodes chacune. Les mesures finales moyennent les graines d’apprentissage sur chaque scénario avant le bootstrap apparié par date. Quatre blocs limitent la portée des intervalles. Les différences entre graines sont conservées dans les métriques finales et les courbes de validation.

## Résultats standard sur 36 scénarios finaux

Les banques comportent 45 scénarios d’entraînement, 12 de validation et 36 de test. Le réglage retenu sur les 12 validations est `α = 0,12`, `γ = 1`. Sa perte moyenne de 1 405,51 MAD est proche des 1 405,65 MAD du second réglage. Les deux obtiennent le même OTIF sur cette validation.

| Politique | OTIF total | OTIF régulier | Urgences servies | Exploitation MAD | Coût avec omissions MAD | Retard min | Changements | Tentatives | Perte MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jamais | 84,17 % | 89,12 % | 0,00 % | 1 277,63 | 1 357,63 | 15,04 | 0,00 | 0,00 | 1 384,69 |
| Événement | 86,79 % | 90,89 % | 19,44 % | 1 280,20 | 1 344,64 | 12,21 | 1,19 | 2,92 | 1 370,47 |
| Périodique | 88,10 % | 91,80 % | 25,00 % | 1 282,72 | 1 342,72 | 10,56 | 1,89 | 25,42 | 1 378,21 |
| Seuil | 87,31 % | 91,64 % | 13,89 % | 1 282,85 | 1 351,74 | 10,57 | 1,53 | 4,64 | 1 376,13 |
| Multicritère | 87,92 % | 91,80 % | 25,00 % | 1 282,44 | 1 342,44 | 10,63 | 1,89 | 8,31 | 1 369,50 |
| Aléatoire disponible | 87,62 % | 91,80 % | 19,44 % | 1 283,36 | 1 347,80 | 10,55 | 1,89 | 20,92 | 1 381,03 |
| Q-Learning | 87,13 % | 91,70 % | 12,04 % | 1 281,17 | 1 351,54 | 10,81 | 1,56 | 25,87 | 1 387,05 |

Face au multicritère, l’agent perd **0,79 point d’OTIF total** (intervalle apparié à 95 % de −1,31 à −0,36) et sa perte est **17,55 MAD plus élevée** (+9,50 à +25,59). Il sert en moyenne 12,04 % des urgences, contre 25 %. Il améliore de 2,96 points l’OTIF par rapport au plan statique (+1,74 à +4,16), mais son écart de perte est de +2,36 MAD (−6,82 à +9,11). Le coût de coordination des tentatives explique une partie de cet écart. La règle aléatoire disponible obtient aussi une perte moyenne inférieure à celle du Q-Learning. Aucun comparateur ne manque de livraison obligatoire. Les quatre dates et les paramètres synthétiques limitent la généralisation de ces intervalles.

Sur les 5 184 décisions des trois graines, l’agent demande 2 794 calculs et applique 67 replans. Le CP-SAT est exécuté 785 fois, dont 70 restrictions locales déclarées infaisables. Aucun repli de sécurité pour candidat invalide ou indisponibilité n’est observé sur le test standard. Les absences ordinaires de candidat restent visibles. La table couvre 31 à 32 états sur 60 et 119 à 122 paires état-action sur 240. Seulement 0,20 à 0,30 % des décisions exploitables de test portent sur un état inédit, selon la graine. Les valeurs et l’arbitrage restent instables malgré cette couverture. L’OTIF final varie de 86,84 % à 87,64 % entre graines, et leurs courbes de validation ne montrent pas une amélioration régulière. Le faible nombre d’états inédits peut coexister avec une perte d’information à l’intérieur de chaque classe.

Sur validation à 20 s, passer de 80 à 40 MAD réduit de trois à une les urgences servies, pour la règle multicritère et pour la première table figée. À 120 MAD, les deux en servent trois, comme à 80 MAD. Entre 5 et 60 s de diffusion, elles en servent toujours trois, et le retard cumulé moyen du Q-Learning passe de 13,31 à 13,70 min. Cette sensibilité porte sur 12 scénarios et une seule table sans réentraînement.

Les 540 épisodes finaux d’apprentissage prennent 149,77 s cumulées, avec 0,277 s par épisode et 6,59 appels CP-SAT en moyenne. Environ 88 % de ce temps passe dans la préparation et la validation des replans. Les 160 épisodes de sélection ajoutent 43,07 s de déroulement d’épisodes. L’inférence tabulaire sur test prend environ 32 µs par décision en moyenne hors calcul de tournée. Le coût principal vient du moteur de dispatch, en particulier de CP-SAT, et le protocole s’exécute localement sur CPU.

## Transfert et limites

Les trois tables standard rechargées affrontent 48 nouveaux scénarios à quatre véhicules, de 22 à 27 clients, sans apprentissage additionnel. L’OTIF est de 89,81 % pour le Q-Learning et 90,25 % pour la règle multicritère. L’écart de perte reste défavorable à l’agent avec +16,40 MAD et un intervalle apparié de +11,74 à +21,06. Le Q-Learning sert 22,22 % des urgences contre 33,33 % pour la règle multicritère. Les tirages de scénario sont nouveaux et les jours de test restent les mêmes. Mesures recherche (historical artifact: `results/experiments/q_learning_research_stress`)

Les écarts reflètent un ensemble synthétique à quatre dates d’évaluation, des pénalités hypothétiques et des services d’urgence souvent rares ou impossibles. L’observation regroupe des tournées avec des marges différentes. L’omission n’est pénalisée qu’en fin de journée, ce qui éloigne certaines conséquences de la décision qui les a provoquées. Les expériences mesurent le résultat de la représentation et du protocole combinés. Elles n’isolent pas la cause de l’écart.

**NO-GO pour un entraînement DQN immédiat.** L’interface et le simulateur sont validés. Avant de changer de modèle, évaluer une observation qui distingue les marges des urgences et un crédit temporel plus proche des décisions, sous les mêmes contraintes et avec des scénarios séparés.

## Exécution et reproduction

Mesures murales et pics RSS de GNU time sur CPU, avec les bibliothèques numériques à deux threads et CP-SAT à un thread. Le profil de recherche exécute ici le transfert des tables standard. Un réentraînement de recherche plus long reste configurable.

| Exécution | Durée | Pic RSS (KiB) |
| --- | ---: | ---: |
| Audit des 33 urgences | 10,06 s | 207 736 |
| Développement | 18,46 s | 233 372 |
| Standard complet | 337,90 s | 287 656 |
| Transfert à quatre véhicules | 84,24 s | 289 184 |
| Notebook Jupyter | 4,17 s | 154 756 |
| 92 tests usuels, dont 8 nouveaux | 9,78 s | 241 748 |
| 35 régressions d’audit | 3,68 s | 148 084 |

Le checker Gymnasium et `pip check` réussissent. Le notebook sauvegarde cinq cellules exécutées et trois figures. Les tables rechargées reproduisent les indicateurs opérationnels des neuf relectures de contrôle.

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -m src.rl.audit
/usr/bin/time -v python -m src.rl.experiment --profile development --output /tmp/smartfmcg-q-development
/usr/bin/time -v python -m src.rl.experiment --profile standard
/usr/bin/time -v python -m src.rl.experiment --profile research --evaluate-checkpoints results/experiments/q_learning
jupyter execute notebooks/08_q_learning.ipynb --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
```

[Configuration](../configs/rl_training.yaml) · Résultats et checkpoints (historical artifact: `results/experiments/q_learning`) · Notebook (historical artifact: `notebooks/08_q_learning.ipynb`)
