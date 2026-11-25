# Phase 7 — Simulation opérationnelle dynamique

## Question scientifique

Une politique de replanification améliore-t-elle le service quand une tournée robuste est déjà en cours, et à quel prix en kilomètres, attente et stabilité ? La comparaison part des séquences et des heures de départ ML avec marge enregistrées dans `0a83bd0`. Le chargement contrôle l’empreinte de ces plans et valide chaque tournée avec le pipeline linéaire et la marge de la phase 6. Toutes les politiques d’un scénario partagent les mêmes départs prévus, stocks et chocs.

## État, événements et conservation

La file d’événements avance à la prochaine date utile en secondes entières depuis minuit. À égalité, les transitions déjà programmées aboutissent, les annonces sont révélées, un plan prêt est vérifié, la politique décide et les nouveaux départs sont lancés. Une jambe est représentée par son origine, sa destination et son heure de départ. La réception d’une alerte de congestion affecte les départs qui suivent, par zone et jusqu’à la fin annoncée. La durée d’une jambe commencée est fixée au départ et reste cachée jusqu’à l’arrivée. Un retard de service révélé avant la fin allonge le service concerné, y compris lorsque le camion est sur place. Une demande urgente crée une commande et un créneau à sa révélation.

Pour chaque dimension, le camion **décharge avant de collecter**. Le stock régulier est affecté à ses commandes et chaque produit de réserve dispose de son propre compartiment. À la fin d’un service, le compartiment pertinent diminue de la livraison et les retours augmentent de la reprise. La somme des compartiments doit rester entre zéro et la capacité véhicule à chaque arrêt. Les véhicules avec tournée peuvent précharger au dépôt jusqu’à huit caisses d’un produit compatible, plafonnées selon la capacité au départ et à chaque arrêt initial. Le journal enregistre le stock chargé et la restitution de la réserve et des retours au dépôt. La capacité disponible sans le produit requis ne suffit pas à accepter une demande.

Exemple testé avec un poids en unités de l’instance miniature. Un camion de capacité 10 part avec 7 unités affectées et 2 en réserve. Il atteint A à t=10. Une annonce à t=15 ajoute 15 s au service qui devait finir à t=20. À t=35, il livre 3 et reprend 4. Le camion porte alors `4 + 2 + 4 = 10` unités. Une réserve de 3 aurait donné 11 après la collecte et le chargement aurait été refusé. Une alerte à t=5 ne change pas l’arrivée déjà engagée à t=10. Elle double le trajet suivant de 10 à 20 s. Dans un second cas, une urgence de 2 unités insérée après B consomme exactement la réserve et le camion rapporte les 4 unités collectées.

## Choix de décision et validation

Une simulation par pas fixe arrondirait les temps et rendrait ambiguë la priorité des événements simultanés. La file fournit ces priorités sans dépendance supplémentaire. Un solveur global à chaque annonce devrait représenter les origines multiples, les charges existantes et la dispersion des produits. Le moteur utilise une insertion faisable et un déplacement d’arrêt par tournée examinée, avec au plus 400 candidats par appel. L’action locale examine le véhicule dont la prévision de retard et de dépassement horaire est la plus forte. L’action globale examine les suffixes de toute la flotte, notamment pour choisir le véhicule qui prend une nouvelle urgence. Une affectation régulière reste liée au stock chargé. Le trajet en cours et le service associé, ainsi que le retour commencé, sont verrouillés.

Le score compare le coût prévisionnel de déplacement et de temps, les retards à 1,80 MAD/min, les dépassements horaires à 3 MAD/min, les changements à 2 MAD par prédécesseur modifié et la pénalité évitée si une urgence est programmée. Le validateur des suffixes contrôle indépendamment couverture, accès, froid, inventaires, capacités et fenêtres prévues des arrêts modifiables. Un candidat doit améliorer le score. Une proposition est revérifiée au moment de la diffusion. Si une nouvelle annonce la rend infaisable, la politique exécutée conserve le plan courant et journalise la cause du refus. Lors du rejeu, la tournée peut subir un retard imprévu malgré ces validations de planification. Le simulateur poursuit les services engagés et mesure les violations horaires réalisées, selon le protocole de phase 6.

Le calcul est borné en nombre de candidats et chronométré en secondes murales. Chaque tentative engagée induit 20 s simulées de diffusion. Les véhicules en mouvement ou en service continuent leurs opérations physiques. Une nouvelle jambe attend la diffusion, même si la tentative ne propose aucun changement. Le coût opérationnel reflète cette attente quand elle intervient entre le départ et le retour. Les métriques indiquent séparément la latence simulée et les secondes-véhicules d’attente réellement provoquées. Le journal indépendant des transitions recalcule les trajets depuis les réalisations et dates de départ, les durées de service, les stocks et les distances à partir des actions physiques.

## Protocole et politiques

Chaque date du 21 au 31 mai 2026 réutilise son plan validé et cinq véhicules actifs sur six, pour 30 commandes régulières. Les réalisations de trafic proviennent du générateur arc-heure de phase 6, sous de nouvelles graines. La couche événementielle tire un incident de zone, un retard de service et une demande urgente sur un site connu. La cible et l’heure du retard sont tirées autour de l’horaire statique, indépendamment de la politique réalisée. Un retard annoncé après un service est sans effet sur ce service. Une seule urgence arrive par scénario. Les prévisions reçoivent le plan à 11 h avec marge, le calendrier et les annonces publiques produites jusque-là.

Chaque politique reçoit les événements et la même horloge de décision avec un point fixe toutes les 15 min. Les règles sont gelées pour la comparaison.

| Politique | Demande d’action |
| --- | --- |
| Jamais | Conserver le plan ML avec marge. |
| À chaque événement | Examiner toute la flotte à chaque annonce. |
| Périodique | Examiner toute la flotte à chaque point fixe. |
| Seuil | Examiner toute la flotte si le retard prévu cumulé atteint 5 min. |
| Multicritère | Examiner toute la flotte si une urgence attend, si un arrêt prioritaire est menacé, si au moins deux arrêts sont à risque ou si le dépassement prévu atteint 3 min. Sinon, examiner un véhicule si un arrêt est à risque et si congestion, retard prévu ou annonce de service franchit le seuil de la règle. La branche locale attend 10 min après la dernière application. |

## Résultats standard

Moyennes sur 44 scénarios appariés, obtenus avec 11 dates, deux graines et deux régimes. L’OTIF compte les commandes livrées en totalité avec fin du service avant ou à la fermeture, sur toutes les commandes, urgences incluses. Le retard additionne les minutes de dépassement des commandes effectivement servies. Une urgence omise coûte 80 MAD dans le modèle. La stabilité mesure les insertions et les changements de prédécesseur sur les suffixes, cumulés au cours de la journée.

| Politique | OTIF total | OTIF régulier | Urgences servies | Exploitation MAD | Coût avec omissions MAD | Retard min | Changements | Tentatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jamais | 88,93 % | 91,89 % | 0/44 | 2 133,75 | 2 213,75 | 12,17 | 0,00 | 0,00 |
| À chaque événement | 90,32 % | 92,35 % | 13/44 | 2 142,09 | 2 198,45 | 10,57 | 0,91 | 3,00 |
| Périodique | 90,98 % | 93,11 % | 12/44 | 2 143,23 | 2 201,41 | 9,60 | 1,48 | 26,70 |
| Seuil | 90,10 % | 92,65 % | 6/44 | 2 137,75 | 2 206,85 | 10,00 | 0,86 | 4,41 |
| Multicritère | 91,13 % | 93,11 % | 14/44 | 2 142,60 | 2 197,14 | 9,59 | 1,59 | 9,05 |

La règle multicritère gagne 2,20 points d’OTIF total par rapport au plan statique. L’intervalle bootstrap à 95 % s’étend de +1,25 à +3,37 points, avec 4 000 rééchantillonnages groupés par date. L’écart d’OTIF régulier est de +1,21 point, avec intervalle de +0,23 à +2,50. Le retard baisse de 2,58 min par jour, avec intervalle de −5,44 à −0,50 min. Le coût avec omissions baisse de 16,61 MAD, avec intervalle de −23,93 à −9,48 MAD. L’exploitation augmente de 8,85 MAD et la distance de 1,36 km. La pénalité d’omission explique l’écart entre les deux mesures de coût. Le périodique atteint le même OTIF régulier moyen avec près de trois fois plus de tentatives que le multicritère. Le déclenchement à chaque événement sert 13 urgences contre 14 pour le multicritère, avec moins de changements et de tentatives. Ces moyennes ne suffisent pas à identifier une politique qui domine les autres sur tous les critères.

Les cinq politiques servent toutes les commandes obligatoires et récupèrent 328 caisses par scénario. Chacune ne réalise qu’une journée sur 44 sans retard, avec zéro dépassement au retour. Le multicritère diffuse en moyenne 0,68 replan par scénario. Ses 9,05 tentatives occupent environ 181 s simulées au total et induisent 7,45 secondes-véhicules d’attente après service. Tous les appels mesurés du portefeuille standard sont plus courts que les 20 s de latence modélisée.

## Robustesse et limites

Dans le profil recherche, 99 scénarios, trois graines différentes et trois régimes donnent au multicritère +2,44 points d’OTIF total, −4,41 min de retard et −12,21 MAD avec omissions contre le statique. Aucune politique ne sert les 33 urgences extrêmes avec le stock et les fenêtres proposés. Dans ce régime, le multicritère réduit néanmoins le retard de 6,34 min et gagne 1,95 point d’OTIF. Son écart de coût moyen de −0,49 MAD est compatible avec zéro dans l’intervalle apparié de −2,49 à +1,25 MAD.

Les 11 dates partagent une demande, une flotte et une géographie synthétiques. Les intervalles comptent 11 blocs de dates et décrivent cette distribution expérimentale. Les résultats sont sensibles aux pénalités, aux hypothèses de réserve, à la visibilité du début et de la fin des alertes et à l’immobilisation au départ des jambes pendant le calcul. Une urgence concerne un site déjà présent dans la matrice. Le simulateur modélise deux références produit, un rechargement initial et un seul retour au dépôt, avec trajets agrégés par arc et sans diversion sur un arc en cours. Les transferts et réapprovisionnements en tournée, les scénarios avec plusieurs urgences et le changement d’échelle spatial restent à étudier. Une recherche locale bornée peut échouer même lorsqu’un ordre est physiquement réalisable. Les retards constatés motivent un objectif adaptatif, avec fenêtres prévisionnelles et capacités contrôlées séparément.

## Exécution locale

Mesures GNU time sur CPU, avec deux threads pour les bibliothèques numériques.

| Exécution | Scénarios et passages | Durée murale | Pic RSS (KiB) |
| --- | --- | ---: | ---: |
| Développement | 3 scénarios, 15 passages | 4,13 s | 207 372 |
| Standard | 44 scénarios, 220 passages | 26,65 s | 208 996 |
| Recherche | 99 scénarios, 495 passages | 54,50 s | 211 596 |
| Notebook Jupyter | 3 cellules, 1 figure, 3 tableaux et texte | 3,77 s | 152 316 |
| Tests usuels | 84 réussis, dont 17 dynamiques | 8,56 s | 226 292 |
| Tests d’audit | 35 réussis | 3,61 s | 147 960 |

Le premier lancement Jupyter dans le bac à sable a été refusé à l’ouverture des sockets du noyau. La relance avec accès local a exécuté et sauvegardé toutes les cellules. `pip check` a confirmé la cohérence des dépendances.

## Reproduction et préparation à la phase 8

`Simulator.next_observation()` avance à la décision suivante, puis `Simulator.act(Action)` reçoit un choix discret. Le moteur n’appelle aucune politique et l’observation ne contient ni les réalisations cachées, ni la file d’événements futurs. Le calcul de perte incrémentale est dans `metrics.reward_delta`. Un futur adaptateur Gymnasium pourra contrôler les pas tout en construisant son observation numérique et son signal de récompense. La formulation de l’observation compacte et son éventuel caractère partiellement observable restent à étudier. Aucun apprentissage RL n’a été lancé.

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
/usr/bin/time -v python -m src.simulation.experiment --profile development --output /tmp/smartfmcg-phase7-development
/usr/bin/time -v python -m src.simulation.experiment --profile standard
/usr/bin/time -v python -m src.simulation.experiment --profile research
jupyter execute notebooks/07_dynamic_simulation.ipynb --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
```

Les commandes lisent les artefacts d’origine et n’ont pas besoin du réseau. Les valeurs simulées et la recherche bornée sont déterministes. Le temps mural et la mémoire varient selon l’exécution. Les profils standard et recherche livrent les métriques, les écarts et intervalles appariés, les événements, les demandes de replan et un journal physique témoin. Résultats standard (historical artifact: `results/experiments/dynamic_simulation`) · Résultats recherche (historical artifact: `results/experiments/dynamic_simulation_research`) · Notebook (historical artifact: `notebooks/07_dynamic_simulation.ipynb`)
