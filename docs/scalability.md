# SmartFMCG · Phase 10.1 · Passage à l’échelle

Le système a été exécuté de bout en bout jusqu’à **100 clients et 16 véhicules disponibles** sur les tables OSRM de Casablanca. Les 138 relectures finales se terminent avec tous les ordres obligatoires livrés et un journal physique valide. À 250 clients, OR-Tools produit des plans sur une géométrie Haversine et dépasse son budget court. À 500 clients, une charge isolée mesure la construction de la matrice et l’inférence. Le défaut provisoire B5 et l’alternative de service B8 restent les politiques de démonstration recommandées. Audit initial · Notebook exécuté (historical artifact: `notebooks/10_1_scalability.ipynb`) · [Données et figures](../results/experiments/scalability)

## Question logistique et protocole

Le passage à l’échelle mesure l’évolution du temps, de la mémoire et du service quand le nombre de clients, la flotte et les perturbations augmentent. Le dimensionnement construit d’abord des tournées témoins avec des véhicules des types historiques. L’objectif de 82 % de charge et la longueur maximale de huit arrêts guident cette construction. Le validateur vérifie les 4 dimensions de chaque charge, le froid, les accès et les horaires réels du plan. Il contrôle de nouveau après déduction de la réserve. Les commandes régulières de l’expérience sont obligatoires. Le service dure 8 minutes plus 1 minute par tranche de six caisses livrées. La fenêtre contrôlée s’étend sur 5 h 30 au maximum et se ferme au plus tard 30 minutes avant le dépôt. La marge prévalidée de phase 6 est appliquée au même tableau de prévision pour les méthodes statiques comparées. Les paramètres de demande et de capacité conditionnent donc l’enveloppe mesurée. [Configuration](../configs/scalability.yaml)

Les cas finaux emploient les 14 et 26 septembre 2026, une journée ouvrée et un samedi, avec la géographie et des tirages exogènes communs aux politiques. Le pilote du 17 août sert au profilage et le 31 août à l’expérience Haversine. Les tailles 30, 50 et 100 emploient deux groupes par date. Le groupe modéré applique des retards plus faibles et charge huit caisses par véhicule destiné à porter une réserve. Le stress augmente la congestion, les retards de service et le nombre d’arrivées urgentes, et réduit la réserve à quatre caisses. Les taux d’arrivées et retards sont configurés par client. À 100 clients, le stress compte 23 événements, dont 10 urgences. Le parc dimensionné va de 6 à 7 véhicules à 30 clients, de 9 à 10 à 50 et de 16 à 100. Les commandes froides tiennent dans un seul véhicule frigorifique par parc dans les cas OSRM mesurés. Le test sans frigorifique force une incompatibilité individuelle et fournit trois cas d’infaisabilité prouvée.

Pour chaque cas, les comparaisons de décision utilisent le plan validé de coût minimal parmi les méthodes essayées. Les véhicules réellement activés portent la réserve chargée avant le départ. Toutes les politiques utilisent ce plan, la même réserve, les mêmes horaires d’événements et les mêmes variations de trafic. Le simulateur révèle ces événements au fil du temps et valide inventaires, positions, déplacements et engagements. L’essai de réserve nulle conserve le plan et le scénario déjà fixés. Pour agréger les agents, la moyenne des trois graines d’apprentissage est calculée au sein de chaque cas, puis les politiques sont comparées par paire. Les figures emploient des étendues descriptives. Deux dates et une seule ville donnent peu de prise pour estimer l’incertitude externe.

## Complexité, faisabilité et budget

Avec \(n\) clients et un dépôt, une matrice a \((n+1)^2\) cellules. À 100 clients, cela donne 10 201 cellules, 10 100 arcs hors diagonale et 242 400 valeurs sur 24 heures. À 50, les 2 550 arcs deviennent presque quatre fois plus nombreux en doublant le nombre de clients. Les tableaux denses ont donc des besoins en temps et en espace au moins quadratiques pour renseigner toutes leurs cellules. Le glouton essaie au plus \(V+k\) positions après avoir placé \(k\) clients sur \(V\) véhicules. Le rejeu d’une position coûte au plus la longueur de la tournée examinée. Si \(L\) borne ces longueurs, l’ordre de grandeur majorant est \(O(n(V+n)L)\). Une relocation sur un suffixe de longueur \(L\) donne jusqu’à \(L(L-1)\) positions dirigées. Le plafond de candidats limite les rejeux et conserve un coût par suffixe. Le HF-VRPSPD-TW est NP-difficile car son cas à un véhicule, avec capacités et fenêtres larges et collecte nulle, contient le problème du voyageur de commerce. [Documentation OR-Tools](https://developers.google.com/optimization/routing)

Avec une capacité de 10 caisses, un véhicule partant avec 9, livrant 3 puis récupérant 5 a \(9-3+5=11\) caisses à bord. Contrôler uniquement le départ aurait accepté une tournée physiquement impossible. Un témoin indépendamment validé établit qu’au moins une solution existe. Une recherche qui s’arrête sans plan sur ce cas révèle une limite de budget ou de construction. Inversement, une commande obligatoire froide avec une flotte entièrement ambiante suffit à prouver l’infaisabilité, même si le reste des tournées est vide.

Le profilage mesure où l’exécution dépense du temps et de la mémoire. Les mesures murales ventilent préparation, caractéristiques, inférence, construction des matrices, simulation et replan. Un profil `cProfile` de développement identifie les appels chauds, avec un surcoût d’instrumentation. La mise en cache réutilise une table dirigée qualifiée sous mêmes coordonnées et provenance. La vectorisation applique un même calcul à un tableau, par exemple en indexant toutes les distances à partir des couples origine–destination. Le traitement par lots borne les tableaux temporaires de caractéristiques. Les prévisions gelées reproduisent les valeurs historiques à \(10^{-12}\) minute près sur les matrices testées. Le démarrage à chaud importe des routes déjà validées dans OR-Tools et laisse le solveur les améliorer. Le délai du premier plan natif est mesuré à partir de l’appel solveur, donc le coût éventuel de la construction du témoin s’ajoute. [Options et statuts OR-Tools](https://developers.google.com/optimization/routing/routing_options) · [Import des routes](https://developers.google.com/optimization/routing/routing_tasks)

## Géographie et prédiction

Les 30 sites historiques sont complétés par des sommets de la géométrie archivée D01–C30, échantillonnés en maximisant leur distance minimale aux sites déjà choisis. Un site ajouté hérite de la zone du site historique le plus proche selon la règle documentée dans le code. Le graphe OSRM automobile provient d’OpenStreetMap. © contributeurs OpenStreetMap. Les données commerciales, les trafics et les événements sont synthétiques. Les sites ajoutés sont concentrés sur un même corridor. Les catégories de trafic du modèle sont donc transférées sous une hypothèse géographique simplifiée. [Attribution OSM](https://www.openstreetmap.org/copyright) · Géométrie archivée (historical artifact: `data/matrices/osrm-D01-C30.geojson`) · [Conditions du service](https://routing.openstreetmap.de/about.html)

| Clients | Cellules | Deux tableaux, Kio | Acquisition OSRM | Relecture cache | Arcs à distance asymétrique |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 961 | 15,0 | archive | 4,17 ms | 96,99 % |
| 50 | 2 601 | 40,6 | 0,54 s, une requête | 6,23 ms | 98,90 % |
| 100 | 10 201 | 159,4 | 20,63 s, trois requêtes | 4,18 ms | 99,13 % |

Les contrôles trouvent zéro arc inaccessible, zéro durée ou distance nulle hors diagonale et zéro raccordement en double. Le nombre de lieux par appel est de 51 pour la table à 50 clients, puis au plus 68 pour les trois tableaux de l’essai à 100. Les requêtes sont espacées d’au moins 1,05 s. La version du graphe n’est pas fournie par le service. Les recouvrements vérifient l’identité des durées, distances et raccordements et les fichiers archivés sont vérifiés par SHA-256. Les 20,63 s sont une observation de réseau et serveur sur une session, avec délais et assemblage. Une étude multi-session serait nécessaire pour estimer la distribution de cette latence. [API de table OSRM](https://project-osrm.org/docs/v26.6.1/http) · Mesures d’acquisition (historical artifact: `results/experiments/scalability/geography_acquisition.csv`)

À 100 clients, 242 400 valeurs sont prévues en environ 1,00 s pour les 24 heures, dont 0,45 s de construction de variables et 0,44 s d’inférence. La conversion de la matrice routière entière prend environ 74 ms et l’application de la marge environ 46 ms. À 500 clients sous Haversine, la prévision traite 6 012 000 lignes en **21,67 s** et atteint un pic de **564,8 Mio** pour le processus. Le profil se limite à la matrice et à la prévision. Les temps de ce test caractérisent le calcul sur les entrées modélisées. La précision en circulation réelle nécessiterait des observations indépendantes. [Prévision et matrices](../results/experiments/scalability/figures/04_prediction_and_matrices.pdf)

## Planification initiale

| Clients | Glouton, cas avec plan | Glouton, temps médian | OR-Tools, cas avec plan à chaque budget et mode | Premier plan à froid / à chaud, médianes à 1,8 s | Coût à froid, médiane à 1,8 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 4/4 | 0,029 s | 4/4 | 0,020 / 0,008 s | 2 026,10 MAD |
| 50 | 4/4 | 0,082 s | 4/4 | 0,045 / 0,022 s | 3 185,30 MAD |
| 100 | 0/4 | 0,264 s | 4/4 | 0,116 / 0,038 s | 6 089,80 MAD |

À 100, le glouton laisse respectivement 6, 6, 5 et 2 visites obligatoires sans affectation malgré la présence d’un témoin. À froid, augmenter le budget de 0,15 à 1,80 s fait passer le coût médian de 6 522,03 à 6 089,80 MAD. À 1,80 s, le démarrage à chaud atteint 6 129,20 MAD de médiane, avec un premier plan détecté plus vite et 0,33 s de dimensionnement du témoin au préalable. L’intérêt opérationnel de l’amorçage est plus direct quand une tournée admissible existe déjà. Tous les plans utilisables sont rejoués. Les statuts de succès de recherche locale donnent des solutions exploitables sous budget. Ils ne prouvent pas un optimum global. À 100 clients, le plan à froid à 1,80 s parcourt une distance médiane de 266,63 km et utilise une médiane de 13,5 véhicules sur 16.

À 250 clients Haversine, les deux appels à 0,15 s expirent. Le callback à chaud détecte un candidat vers 0,147 s, puis le budget du processus de recherche est épuisé avant retour d’un plan exploitable. Le témoin indépendant reste viable. À 0,65 s, les deux appels renvoient un plan validé avec 39 véhicules disponibles et des coûts de 16 233,78 MAD à froid et 17 250,70 MAD à chaud. Le glouton et la simulation sont hors de ce profil de charge. L’essai à une seule date informe sur les limites du budget et mérite peu de généralisation. [Temps de routage](../results/experiments/scalability/figures/01_initial_routing.pdf) · [Qualité par budget](../results/experiments/scalability/figures/03_solution_quality.pdf) · [Plans et statuts](../results/experiments/scalability/planning_summary.csv)

## Décisions, budgets et service réalisés

| Politique, 100 clients | Perte simulée moyenne (MAD) | OTIF total | Urgences servies | Temps cumulé de replan par journée |
| --- | ---: | ---: | ---: | ---: |
| B4, maintien | 6 917,92 | 86,10 % | 0,00 % | 0,00 s |
| B5, événementielle | 6 828,52 | 90,06 % | 6,25 % | 0,81 s |
| B8, multicritère | 6 836,29 | 90,51 % | 6,25 % | 1,54 s |
| B9, Q-Learning | 6 803,83 | 91,66 % | 12,50 % | 1,00 s |
| B10, DQN | 6 865,35 | 89,22 % | 4,17 % | 0,48 s |

Les valeurs à 100 clients agrègent quatre cas, à 400 candidats. B5 obtient en moyenne la perte la plus faible parmi les règles aux trois tailles. B8 obtient un OTIF moyen légèrement supérieur à 30 et 100 clients. À 100 clients, B9 réduit en moyenne la perte de 24,69 MAD par rapport à B5, avec un écart par cas de −44,40 à +13,00 MAD et 10,4 appels CP-SAT par journée en moyenne. B9 perd face à B5 à 30 et 50 clients. L’effet observé à 100 justifie de garder B9 comme comparateur. Le nombre de dates, l’étendue des écarts et les besoins de calcul réduisent la portée du classement. B10 utilise surtout le maintien et l’action globale, avec environ 0,54 ms d’inférence par décision à 100 clients. B10 gagne face à B4 en moyenne à chacune des tailles. Il est dominé par B5 en perte moyenne aux trois tailles. [Performances appariées](../results/experiments/scalability/figures/08_policy_transfer.pdf) · [OTIF par régime](../results/experiments/scalability/figures/07_otif.pdf)

En stress à 50 clients, de 100 à 2 000 candidats donnent le même service et la même perte dans les deux cas pour B5 et B8. À 100 clients, le passage de 100 à 400 réduit la perte moyenne de 30,79 MAD pour B5 et de 13,45 MAD pour B8, tout en augmentant respectivement l’OTIF de 1,36 et 0,45 point. À 1 000 et 2 000, le travail supplémentaire ne change ni perte, ni service, ni itinéraires observés dans ces deux cas. Pour B8, passer de 400 à 1 000 fait passer le temps cumulé moyen d’environ 1,96 à 3,75 s. Les politiques B5 et B8 ne servent aucune des urgences des deux cas stressés à 100 clients. Retirer les 44 caisses embarquées dans ces cas ne change donc pas leurs résultats. Cette ablation compare le stock à itinéraire et événements fixés, dans deux cas sans service urgent sous l’approche retenue. L’effet d’une réserve mieux localisée ou plus grande reste à mesurer. [Budgets de candidats](../results/experiments/scalability/figures/06_candidate_budgets.pdf) · [Ablation réserve](../results/experiments/scalability/reserve_ablation.csv)

La latence de diffusion modélisée est de **20 s simulées par tentative**. La mesure murale couvre le calcul et la validation. À 100 clients et 400 candidats, le P95 regroupé des tentatives vaut 17,47 ms pour l’heuristique locale, 58,44 ms pour le replan global et 85,75 ms pour le sous-problème local CP-SAT. Le maximum des P95 par journée vaut 147 ms, toutes variantes OSRM à 100 incluses. Les appels CP-SAT sans plan admissible sont enregistrés à part des replis de sécurité. Le moteur apparié construit la sonde d’observation même pour les règles, afin de partager le même chemin d’exécution. Les comparaisons totalisent 108 relectures principales, 24 sur la sensibilité des budgets et 6 pour la réserve. Aucun ordre obligatoire omis, aucune violation physique, aucun repli de sécurité et aucune action hors masque dans ces relectures. [Temps par action](../results/experiments/scalability/replanning_action_profile.csv) · [Temps de replan](../results/experiments/scalability/figures/05_replanning_time.pdf) · [Décomposition](../results/experiments/scalability/figures/09_runtime_breakdown.pdf)

## Transfert, limites et décision

La généralisation hors distribution désigne l’emploi d’une représentation apprise sur des états dont les caractéristiques changent. À 100 clients, **567 des 598 décisions éligibles DQN, soit 94,8 %**, dépassent au moins une plage observée dans le tampon correspondant à leur graine. À 30 clients, c’est déjà 401/453, soit 88,5 %, ce qui illustre le décalage de la génération et des réserves entre phases. Le nombre de véhicules flexibles, les caisses ambiantes disponibles et les arrêts modifiables dominent à 100 clients. Le vecteur de 22 composantes, les transformations, le masque et les politiques sont appliqués tels que gelés. Les minima et maxima archivés fournissent un indicateur univarié sensible. Ils ne représentent ni la densité jointe, ni une probabilité d’échec. L’agent tabulaire rencontre 0 état éligible inédit sur 588 à 100 clients, avec un état inédit sur 452 à 30. Sa discrétisation peut rapprocher des situations physiques très différentes. [Couverture](../results/experiments/scalability/observation_support.csv) · Distributions d’action (historical artifact: `results/experiments/scalability/action_counts.csv`)

| Exécution | Temps dans le moteur | Pic RSS |
| --- | ---: | ---: |
| OSRM 30, quatre cas | 35,99 s | 534,4 Mio |
| OSRM 50, quatre cas | 53,71 s | 545,6 Mio |
| OSRM 100, quatre cas | 111,97 s | 566,5 Mio |
| Haversine 250, planification et prévision | 11,26 s | 485,6 Mio |
| Haversine 500, matrice et prévision | 22,27 s | 564,8 Mio |

Les mesures proviennent du Core i5-1345U, Python 3.14.2 et deux threads numériques. Elles incluent les objets chargés dans le processus. Les profils Haversine n’ouvrent pas les modèles RL. L’échantillonnage RSS par relecture et le pic du processus fournissent deux vues de la mémoire. Dans le pilote instrumenté, la projection répétée des suffixes, les opérations de charge et les appels CP-SAT dominent parmi les fonctions du replan. La boucle `IndexToNode` d’OR-Tools et les validations des tables du simulateur apparaissent aussi dans les mesures. Dans l’essai 500, les caractéristiques et l’inférence dominent. Profil CPU (historical artifact: `results/experiments/scalability/python_profile_top.csv`) · [Mémoire](../results/experiments/scalability/figures/02_peak_memory.pdf)

Le test de stress change plusieurs facteurs de perturbation simultanément. La bande de géographie routière, les fenêtres construites avec marge pour planifier, les camionnettes modélisées et deux dates limitent les conclusions sur un réseau plus varié, des tournées massivement plus longues ou davantage de véhicules frigorifiques. L’étape au-delà de 100 clients nécessiterait une géographie routière plus diversifiée et un OSRM dédié sur un extrait OSM daté, une qualification des raccordements, une table versionnée et un contrôle des budgets avant simulation. Les échéances de 5 000 candidats et de 1 000 clients n’ont pas été exécutées. Les mesures CPU et mémoire ne motivent pas d’infrastructure GPU ou distante pour le périmètre testé.

**GO recommandé pour la phase 11**, dans l’enveloppe de démonstration mesurée, sous approbation explicite. ALNS détruit une partie d’une solution puis la répare en choisissant des opérateurs dont les poids évoluent d’après les résultats. Elle peut explorer des réaffectations plus larges que des relocations isolées, au prix de nouveaux opérateurs, d’un réglage et de contrôles de charge et de fenêtres après réparation. Un hybride pourrait amorcer ALNS avec un plan OR-Tools. Une évaluation incrémentale pourrait accélérer les projections répétées tout en comparant chaque admissibilité au validateur existant. L’élagage des candidats exige de quantifier les insertions utiles perdues. Une décomposition spatiale réduit la recherche mais peut mal répartir les véhicules froids et la réserve aux frontières. Le gain au-delà de 400 candidats est nul dans les essais ciblés et OR-Tools fournit des plans solides jusqu’à 100. Ces observations ne justifient pas une phase 10.2 avant le tableau de bord. [Røpke et Pisinger, 2006](https://doi.org/10.1287/trsc.1050.0135)

## Validation et reproduction

Les 132 tests usuels et les 35 tests d’audit réussissent, ainsi que `pip check`. Les nouvelles vérifications couvrent la reproductibilité, la croissance des matrices, les témoins, le froid, les rapprochements du cache, la parité des prévisions et le contrat des politiques gelées. Le notebook final exécute ses cinq cellules de code et rend les neuf figures PDF et PNG 300 dpi. Son exécution prend 3,08 s et atteint 107,1 Mio de RSS. Le jalon est identifié par le message Git indiqué ci-dessous. Les prévisions, les historiques et les agents gelés se rechargent sur CPU.

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-dqn.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
out=$(mktemp -d /tmp/smartfmcg-101.XXXXXX)
export MPLCONFIGDIR="$out/mpl"
for n in 30 50 100; do
  /usr/bin/time -v timeout 195 python -m src.scalability.experiment \
    --profile standard --customers "$n" --output "$out/standard_$n"
done
for n in 100 250 500; do
  /usr/bin/time -v timeout 135 python -m src.scalability.experiment \
    --profile proxy --customers "$n" --output "$out/proxy_$n"
done
python -m cProfile -o "$out/development.prof" -m src.scalability.experiment \
  --profile development --customers 30 --output "$out/development_30"
cp results/experiments/scalability/geography_acquisition.csv "$out/"
python -m src.scalability.analysis --output "$out" --cprofile "$out/development.prof"
python -m src.scalability.figures --output "$out"
cp notebooks/10_1_scalability.ipynb "$out/"
SMARTFMCG_SCALABILITY_OUTPUT="$out" jupyter execute "$out/10_1_scalability.ipynb" --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
git log -1 --oneline --grep='^Validate SmartFMCG scalability and stress tests$'
```

Pour chronométrer la relecture de la table qualifiée à 100, exécuter avec `--cache-only`. Les données temporelles d’acquisition initiale restent dans le CSV historique de la phase. Les temps muraux du solveur peuvent changer le meilleur plan retenu selon la machine. Les cas, événements, graines, matrices et tournées retenues sont enregistrés dans les fichiers `cases.json`, `matrix.json` et `protocol.json` de chaque répertoire de travail.

```bash
python -m src.scalability.geography --customers 100 \
  --endpoint https://routing.openstreetmap.de/routed-car \
  --cache data/scalability --max-locations 68 --block-size 34 \
  --graph fossgis-2026-11-25-session --cache-only
```
