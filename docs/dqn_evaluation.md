# Phase 9B — Entraînement et évaluation scientifique du DQN

## Question opérationnelle et formulation

La phase part du checkpoint propre `a6968e5`, qui valide la phase 9A. En exploitation, une replanification peut sauver une commande urgente ou éviter un retard, au prix d’une coordination, d’une attente de diffusion et de modifications de tournée. Une urgence livrée évite 80 MAD d’omission. Avec 12 MAD de coût d’exploitation supplémentaire, un changement à 2 MAD et une tentative à 0,50 MAD, le gain est de 65,50 MAD si le reste de la journée est identique. La comparaison mesure la qualité de cet arbitrage.

Le simulateur avance d’une décision à la suivante. Le processus est partiellement observable, car le réseau voit des indicateurs publics de l’heure, de l’éligibilité, des engagements, du risque et des urgences, tandis que le moteur conserve le trafic réalisé et les événements à venir. Le code compact utilise les classes d’origine, encodées à la même largeur de 22 entrées. L’encodage continu normalise 22 indicateurs par des constantes déjà fixées en phase 8.1. Les deux réseaux de 5 892 paramètres ont deux couches ReLU de 64 neurones. La table permet d’estimer une valeur par paire état–action. Le DQN partage ses paramètres entre états et peut exploiter des marges continues, avec une approximation et un calcul supplémentaires.

La cible à un pas s’écrit `y = r + γ (1 − d) max Q_cible(o′, a′)`, avec `a′` pris dans le masque public et `d = 1` à la fin réelle. Le simulateur valide les actions de maintien, d’heuristique locale, de CP-SAT local et de recherche globale, ainsi que les plans candidats et les mouvements réalisés. Au-delà de l’éligibilité, le calcul peut échouer, être refusé ou n’apporter aucune amélioration. La troncature externe conserve la continuation dans le rejeu. Exemple avec `r = −2`, `γ = 1`, `Q_cible(maintien) = 2` et une sortie de 40 pour une action masquée. La cible vaut 0 si la prochaine décision permet seulement le maintien et −2 si la journée se termine.

SB3 2.8.0 fournit Adam, la perte de Huber, les lots du tampon et la mise à jour du réseau cible. Dans l’implémentation du projet, le masque filtre les tirages pendant l’échauffement et l’exploration, l’argmax glouton, la cible de Bellman et l’évaluation déterministe. Le maintien reste disponible. Le bit d’éligibilité représente la possibilité de lancer un calcul à partir de la situation publique. La validation indépendante contrôle les décisions physiques, y compris les engagements, stocks, capacités sur quatre dimensions, fenêtres et compatibilités. La perte inclut les retards réellement constatés. [Code de SB3](https://stable-baselines3.readthedocs.io/en/v2.8.0/_modules/stable_baselines3/dqn/dqn.html) · [Masque et adaptateur](../src/rl/dqn.py)

## Diagnostic du maintien systématique en phase 9A

| Constat mesuré | Interprétation et limite |
| --- | --- |
| 543 et 483 maintiens sur deux tampons de 768 transitions. Parmi les seules décisions éligibles, 204/429 et 165/450 maintiens. Toutes les interventions ont été échantillonnées. | Une grande part du déséquilibre brut vient du masque. Le nombre d’interventions enregistrées ne renseigne pas à lui seul sur leur qualité. |
| 60 et 57 transitions éligibles avec urgence en attente. La sonde publique trouve une insertion pour au moins une demande sur 8 et 23 de ces transitions. | L’agent reçoit peu d’exemples de cette situation. L’échec de la sonde bornée peut coexister avec une solution qui réordonne les visites. |
| Parmi les maintiens éligibles, environ 40 à 41 % des récompenses immédiates sont nulles. Dans les deux tampons, les 30 transitions terminales sont toutes des maintiens inéligibles. | Une intervention peut acquitter des frais immédiats et éviter une omission bien après la décision. Les récompenses par action mélangent des états différents. |
| Les interventions dont la journée se termine dans le rejeu sont à environ 33–36 décisions de cette fin. À `γ = 1`, la pénalité a le même poids quelle que soit sa distance. À 0,995, elle pèserait environ 84 % à 34 décisions. | L’actualisation ne peut pas expliquer l’annulation du bénéfice dans le pilote. La propagation de ce bénéfice par cibles à un pas reste une difficulté plausible. |
| Les médianes des Q joués éligibles sont entre −2,61 et −1,90 MAD, alors que les écarts moyens aux retours observés des interventions sont de 588 à 716 MAD selon la graine et l’action. | L’ordre de grandeur des retours à long terme est très éloigné de celui des Q. Ces retours viennent du comportement d’exploration et dépendent de ses choix futurs. Ils ne permettent pas d’estimer un biais par rapport à la valeur optimale. |
| Aux checkpoints finaux du pilote, les deux réseaux s’accordent avec leurs réseaux cibles sur l’action maximale dans tous les états suivants éligibles du tampon. | Le biais du maximum reste possible en théorie. Le diagnostic ne relève aucun désaccord d’argmax à ces checkpoints. Un accord ponctuel n’établit pas la stabilité de l’entraînement. |
| Sur les 16 scénarios de diagnostic, 10 des 146 décisions échantillonnées admettent un gain sous intervention isolée. Huit sur 75 avec urgence en attente et deux sur 71 sans urgence. Le maximum atteint 74,50 MAD. | Les opportunités utiles sont rares dans cet échantillon. Les décisions d’un même scénario sont dépendantes. La sélection examine toutes les décisions éligibles avec urgence et une décision ordinaire sur six. Les fractions dépendent de ce plan d’échantillonnage. |

Pour les bifurcations, le multicritère fournit une trajectoire de départ commune. On copie le simulateur à une décision éligible, on teste chacune des quatre actions avec la même information publique, puis on poursuit avec maintien jusqu’à la fin. Le gain est la différence de perte face au maintien dans la même copie. Les résultats futurs servent exclusivement à attribuer le gain rétrospectif. Les deux réseaux continus de phase 9A gardent tous les plans à ces points. Le taux 0,001 demande 35 interventions, dont six à gain positif et 29 à perte supérieure, pour un gain moyen de 2,29 MAD par décision échantillonnée. Cette continuation fixe ne permet pas d’identifier une politique séquentielle optimale.

Les observations résument les tournées et les demandes. Parmi les 21 classes d’origine visitées dans les bifurcations, cinq rassemblent à la fois des décisions à gain strictement positif et des décisions sans gain. Avec l’encodage tabulaire raffiné, trois des 24 classes sont dans ce cas. Les deux encodages cachent des différences de créneaux ou de clients. Même le vecteur continu ignore une partie des états physiques et les réalisations futures. Les différences observées ne permettent pas d’attribuer le maintien à une seule cause. Le volume d’interactions, la décroissance d’exploration, la fréquence des urgences, l’échelle en MAD, le choix du taux et la représentation peuvent interagir. Les cibles utilisent une perte de Huber de seuil 1 MAD. Sur les deux tampons de phase 9A, 274 et 267 récompenses dépassent 1 MAD en valeur absolue. Il est plausible que la taille des retours et la Huber limitent la vitesse d’ajustement des grandes erreurs. L’expérience conserve l’échelle en MAD et mesure l’erreur TD au fil des mises à jour. Analyse des tampons (historical artifact: `results/experiments/dqn_evaluation/pilot_replay_analysis.csv`) · Bifurcations (historical artifact: `results/experiments/dqn_evaluation/diagnostic_forks.csv`) · [Biais du maximum dans DQN](https://arxiv.org/abs/1509.06461)

## Présélection et revue avant extension

Seuls les deux taux proposés changent dans le premier essai continu. Même initialisation issue de la graine 2903, même affectation des scénarios, `γ = 1`, tampon de 1 024, lot de 32, échauffement de 96 pas, gradient toutes les quatre interactions et synchronisation cible toutes les 128 interactions. Le calendrier ε passe de 0,90 à 0,08 sur les 1 600 premières interactions. Pendant les 96 premières, l’échauffement impose le tirage uniformément masqué. L’échelle de décroissance en interactions est conservée dans l’entraînement prolongé. Les checkpoints à 1 000 et 2 000 incluent toutes les mises à jour jusqu’à ces étapes.

| Taux | Interactions | Perte de validation (MAD/jour) | OTIF total | Urgences servies | Tentatives/jour | Appels CP-SAT/jour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,0003 | 1 000 | 1 438,91 | 81,14 % | 0,00 % | 0,00 | 0,00 |
| 0,0003 | 2 000 | 1 438,91 | 81,14 % | 0,00 % | 0,00 | 0,00 |
| 0,001 | 1 000 | 1 440,24 | 81,66 % | 0,00 % | 2,67 | 2,42 |
| 0,001 | 2 000 | 1 428,06 | 81,63 % | 16,67 % | 1,83 | 1,67 |

Les deux essais ont 1 209 transitions éligibles sur 2 000. Leurs demandes d’intervention sur ces transitions diminuent entre les 500 premiers et les 500 derniers pas, de 72,9 % à 28,9 % et de 64,8 % à 23,1 %. Sur la dernière tranche, les pertes TD médianes par mise à jour sont de 26,58 et 26,72. L’écart de norme L2 entre les deux réseaux au checkpoint de 2 000 est de 0,33 et 0,34, après 15 synchronisations chacun. La différence de qualité opérationnelle n’est donc pas reflétée par ces seules mesures d’optimisation. Le taux 0,001, meilleur de 10,84 MAD sur validation, est retenu sur une seule graine. Aucun réglage de la récompense ni de l’architecture n’est motivé par cette revue. Revue enregistrée (historical artifact: `results/experiments/dqn_evaluation/review.json`)

## Protocole final

La banque compte 36 tirages d’apprentissage à trois véhicules, 12 de validation et 16 de diagnostic. Ces rôles utilisent quatre, deux et deux dates distinctes. Les scénarios finaux sont tirés après gel des modèles, sur les trois dates restantes, avec 18 cas à trois véhicules et 18 à quatre véhicules. Avant de générer ces cas, le profil CPU a conduit à retenir six tirages par couple date–flotte sur les huit configurés comme plafond. Cela maintient deux tirages par régime dans chaque couple et réserve le temps nécessaire au rejeu de tous les comparateurs. Les cinq dimensions varient conjointement avec le scénario. Il s’agit de la demande régulière, de la flotte, des réserves, des heures et intensités des perturbations et des caractéristiques d’une ou deux urgences. Les cas à trois véhicules couvrent 15 à 20 clients et 494 à 604 caisses régulières, ceux à quatre véhicules 22 à 27 clients et 665 à 765 caisses. Onze des 36 cas comportent deux urgences. Les graines sont disjointes des phases 8, 8.1 et 9A. Les dates de référence restent dans la même bibliothèque synthétique et sont réutilisées entre phases. Toutes les politiques reçoivent les mêmes scénarios, contraintes, annonces et horloges de décision. [Configuration](../configs/dqn_evaluation.yaml) · [Protocole gelé](../results/experiments/dqn_evaluation/frozen_protocol.json)

Les deux DQN et les tables A, B et D sont entraînés sur les graines 2917, 2927 et 2953, avec 4 000 interactions par paire variante–graine. A reprend l’observation et le retour d’origine, B utilise la discrétisation raffinée et D ajoute l’imputation anticipée de l’omission prouvée. La somme non actualisée reste réconciliée à la perte de phase 8. Les six règles gelées comprennent le maintien, le déclenchement aux événements éligibles, le périodique, le seuil, le multicritère et le tirage uniforme masqué. Les modèles chargés en référence incluent les tables de phase 8, les variantes A à D de 8.1 et les deux encodages du pilote 9A. L’apprentissage historique a son propre budget. Le comparateur aléatoire utilise une graine fixe.

Pour chaque nouvelle combinaison, le checkpoint entraîné minimise la moyenne de perte sur validation. L’encodage DQN de référence minimise la moyenne des pertes de validation des checkpoints sélectionnés sur les trois graines. Les résultats conservent toutes les graines et indiquent la dispersion. Quatre mille tirages bootstrap rééchantillonnent les dates, puis les cas appariés dans chaque couple date–flotte, puis les graines. Les nouveaux modèles partagent les tirages de graines et les historiques ont leurs propres tirages. Les intervalles avec seuls blocs de dates figurent également dans les résultats. Trois blocs limitent fortement la portée des intervalles.

### Sélection des checkpoints DQN

| Encodage | Graine | Checkpoint | Perte de validation (MAD/jour) | Tentatives/jour |
| --- | ---: | ---: | ---: | ---: |
| Continu | 2917 | 2 000 | 1 421,70 | 0,50 |
| Continu | 2927 | 4 000 | 1 428,56 | 2,83 |
| Continu | 2953 | 3 000 | 1 429,06 | 4,42 |
| Compact | 2917 | 1 000 | 1 441,80 | 29,08 |
| Compact | 2927 | 3 000 | 1 432,40 | 10,33 |
| Compact | 2953 | 3 000 | 1 429,10 | 15,08 |

Les moyennes par encodage sont de 1 426,44 et 1 434,43 MAD. Le continu est donc la référence avant le test. Les trois continus gardent tous les plans à 1 000 interactions et interviennent ensuite aux checkpoints retenus. Le continu de graine 2917 revient au maintien à 4 000 interactions. Le nombre d’interactions est lié à l’évolution de la politique sur ces courbes, sans amélioration monotone. Le passage du pilote à 9B combine aussi un taux, un calendrier d’exploration et des scénarios différents.

## Coût d’apprentissage sur CPU

| Groupe | Interactions | Mises à jour réseau | Appels CP-SAT | Temps d’apprentissage (s) | Validation et sauvegarde (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Présélection DQN | 4 000 | 952 | 401 | 29,32 | 21,05 |
| DQN continu | 12 000 | 2 928 | 729 | 72,51 | 25,18 |
| DQN compact | 12 000 | 2 928 | 1 412 | 106,57 | 40,24 |
| Tabulaire A | 12 000 | 0 | 1 660 | 79,91 | 43,36 |
| Tabulaire B | 12 000 | 0 | 1 571 | 84,63 | 50,05 |
| Tabulaire D | 12 000 | 0 | 1 572 | 71,12 | 35,31 |

Les entraînements totalisent 28 000 interactions et 6 808 mises à jour de réseau pour le DQN, ainsi que 36 000 interactions et mises à jour tabulaires. La planification consomme environ 313 s d’apprentissage cumulé sur 444 s. Les mises à jour réseau prennent 30,64 s au total, soit environ 4,50 ms par mise à jour. La représentation intervient aussi via les actions qu’elle entraîne. Par exemple, les trois DQN compacts lancent 1 412 appels CP-SAT en apprentissage, contre 729 pour les continus. Les deux bibliothèques de calcul utilisent le CPU, avec deux threads pour les opérations numériques et un thread pour chaque appel CP-SAT. Journal cumulatif (historical artifact: `results/experiments/dqn_evaluation/budget.json`)

## Test final et interprétation

Les chiffres sont des moyennes par scénario et, pour les agents, par graine. Le service urgent compte les commandes livrées et l’OTIF exige une livraison complète et ponctuelle. Le coût avec omissions précède la pénalisation du retard, du dépassement, de la coordination et de l’instabilité. Les KPIs détaillés et les distributions par action sont disponibles pour les 1 440 journées rejouées. Mesures par scénario (historical artifact: `results/experiments/dqn_evaluation/final_cases.csv`) · [Demandes et exécutions par action](../results/experiments/dqn_evaluation/final_action_distribution.csv)

| Politique | Perte (MAD) | OTIF total | OTIF régulier | Urgences servies | Exploitation (MAD) | Avec omissions (MAD) | Retard (min) | Tentatives | Replans appliqués | Changements | CP-SAT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Événementielle | 1 582,04 | 87,67 % | 91,47 % | 26,39 % | 1 479,03 | 1 554,58 | 13,05 | 3,28 | 0,56 | 1,17 | 0,00 |
| Multicritère | 1 584,52 | 87,99 % | 91,81 % | 26,39 % | 1 479,22 | 1 554,77 | 12,68 | 8,50 | 0,64 | 1,33 | 0,00 |
| DQN continu 9B | 1 588,95 | 86,93 % | 91,12 % | 18,98 % | 1 477,52 | 1 560,48 | 14,22 | 2,65 | 0,37 | 0,78 | 0,00 |
| À seuil | 1 593,05 | 87,13 % | 91,57 % | 13,89 % | 1 476,80 | 1 565,69 | 12,89 | 4,78 | 0,39 | 0,89 | 0,00 |
| Périodique | 1 593,45 | 87,46 % | 91,37 % | 26,39 % | 1 478,94 | 1 554,49 | 13,10 | 26,00 | 0,58 | 1,19 | 0,00 |
| Tabulaire A 9B | 1 594,29 | 87,24 % | 91,13 % | 23,61 % | 1 478,70 | 1 556,48 | 14,53 | 19,45 | 0,45 | 0,96 | 5,88 |
| Tabulaire C 8.1 | 1 596,27 | 87,15 % | 91,40 % | 18,52 % | 1 477,57 | 1 561,27 | 13,07 | 19,03 | 0,46 | 0,98 | 9,98 |
| Aléatoire masqué | 1 596,54 | 87,33 % | 91,40 % | 22,22 % | 1 477,57 | 1 557,57 | 14,32 | 22,06 | 0,53 | 1,08 | 6,81 |
| Tabulaire A 8.1 | 1 596,68 | 87,20 % | 91,42 % | 20,83 % | 1 477,11 | 1 559,33 | 13,47 | 22,07 | 0,50 | 1,03 | 2,88 |
| DQN compact 9B | 1 597,79 | 87,22 % | 91,62 % | 17,13 % | 1 477,30 | 1 562,49 | 13,32 | 18,57 | 0,49 | 1,02 | 3,25 |
| Tabulaire D 9B | 1 600,95 | 86,64 % | 91,50 % | 9,72 % | 1 476,49 | 1 569,82 | 13,05 | 12,22 | 0,35 | 0,76 | 2,06 |
| Q-Learning phase 8 | 1 601,39 | 87,15 % | 91,63 % | 17,59 % | 1 476,61 | 1 562,54 | 12,96 | 27,06 | 0,48 | 1,00 | 7,14 |
| DQN compact 9A | 1 604,22 | 85,69 % | 90,99 % | 0,69 % | 1 472,69 | 1 576,03 | 14,77 | 2,36 | 0,10 | 0,21 | 0,00 |
| Tabulaire B 9B | 1 604,90 | 86,74 % | 91,61 % | 10,19 % | 1 476,97 | 1 569,56 | 12,97 | 20,58 | 0,39 | 0,85 | 7,44 |
| Tabulaire B 8.1 | 1 604,93 | 86,73 % | 91,61 % | 10,19 % | 1 475,16 | 1 568,49 | 12,99 | 22,90 | 0,39 | 0,81 | 12,66 |
| Tabulaire D 8.1 | 1 605,21 | 86,59 % | 91,33 % | 12,04 % | 1 475,24 | 1 567,09 | 13,63 | 24,32 | 0,34 | 0,71 | 12,61 |
| DQN continu 9A | 1 605,82 | 85,33 % | 90,70 % | 0,00 % | 1 472,26 | 1 576,71 | 16,17 | 0,00 | 0,00 | 0,00 | 0,00 |
| Maintien | 1 605,82 | 85,33 % | 90,70 % | 0,00 % | 1 472,26 | 1 576,71 | 16,17 | 0,00 | 0,00 | 0,00 | 0,00 |

Le continu sélectionné sur validation obtient 86,93 % d’OTIF total et 18,98 % d’urgences servies, contre 85,33 % et 0 % pour le maintien. Il réduit la perte moyenne du maintien de 16,87 MAD. Il sollicite 2,65 replans par scénario, avec 0,37 replan effectivement appliqué. L’action CP-SAT locale n’est demandée dans aucun de ses 108 rejeux. Le DQN compact demande 18,57 replans et 3,25 appels CP-SAT par scénario. Il atteint un OTIF de 87,22 % et une perte supérieure de 8,85 MAD. Le continu obtient aussi une perte inférieure aux trois tables à budget comparable, avec une qualité de service différente selon la table. Face à A, il sert moins d’urgences et évite 16,81 tentatives par scénario.

Les règles gardent un avantage opérationnel moyen. La règle événementielle obtient 1 582,04 MAD de perte et 26,39 % d’urgences servies. Le multicritère obtient 1 584,52 MAD, 87,99 % d’OTIF et 26,39 % d’urgences servies. Le DQN continu atteint 1 588,95 MAD. Le surcoût face au multicritère est de 4,43 MAD, avec 1,06 point d’OTIF et 7,41 points de service urgent en moins. Les règles disposent d’un déclencheur explicite à partir des annonces et des données de risque. Leur avantage est compatible avec un problème où les interventions utiles sont rares et où l’information publique porte déjà des indicateurs métier forts.

### Dispersion et transfert

Les trois graines du continu donnent entre 1 584,38 et 1 596,35 MAD de perte moyenne et entre 8,33 % et 25,00 % d’urgences servies. La graine 2917 propose 0,42 tentative par scénario et sert 25,00 % des urgences, tandis que 2927 propose 3,61 tentatives et en sert 8,33 %. Dans cet essai, le volume de tentatives renseigne peu sur la qualité du déclenchement. Les trajectoires de validation et cette dispersion indiquent une sensibilité à la graine et au checkpoint.

À trois véhicules, la perte du DQN continu est de 1 371,82 MAD, contre 1 382,48 pour le maintien et 1 368,33 pour la règle événementielle. À quatre véhicules, elle atteint 1 806,08 MAD, contre 1 829,15 et 1 795,76 MAD. Le gain par rapport au maintien se retrouve donc sur les deux tailles de flotte. L’écart défavorable face à la règle événementielle augmente dans le transfert. Les flottes de transfert restent issues de la même bibliothèque de Casablanca et les contraintes, produits et mécanismes d’événement utilisent le même générateur. Les tests renseignent la robustesse à la taille de flotte dans ce cadre synthétique.

### Écarts appariés et coût d’inférence

Sur les mêmes 36 cas, l’écart de perte du DQN continu vaut −16,87 MAD face au maintien, avec un intervalle bootstrap descriptif à 95 % de [−33,71 ; −4,67]. Il vaut +6,90 MAD [0,39 ; 16,00] face à la règle événementielle et +4,43 MAD [−1,83 ; 14,11] face au multicritère. L’OTIF du continu perd 1,06 point [−1,91 ; −0,40] face au multicritère. Face au compact, l’écart de perte est de −8,85 MAD [−21,96 ; 1,23], et face à la table A de −5,34 MAD [−13,42 ; 3,97]. Ces deux comparaisons restent ouvertes compte tenu de leur dispersion. L’avantage face aux tables B et D ressort avec −15,95 MAD [−26,14 ; −7,94] et −12,00 MAD [−21,71 ; −4,62]. Les deux DQN continus du pilote retrouvent exactement le maintien sur cette nouvelle banque. [Intervalles complets et sensibilité aux seuls blocs de dates](../results/experiments/dqn_evaluation/paired_intervals.csv)

Le continu demande 130 heuristiques locales et 156 recherches globales, qui donnent respectivement 11 et 29 replans appliqués. Toutes les politiques confondues, les 1 440 relectures donnent zéro demande invalide, zéro repli de sécurité physique et zéro commande obligatoire omise. Les 727 appels CP-SAT sans solution faisable retournée sont enregistrés à part, avec le statut et le contexte du sous-problème. Le modèle continu consomme en moyenne 0,803 ms par décision d’inférence, hors calcul de tournée, le compact 0,537 ms, la table A 0,034 ms et le multicritère environ 0,002 ms. Du fait de ses appels au solveur, la table A est plus lente sur une journée complète que le DQN continu, 0,294 s contre 0,119 s. La règle événementielle et le multicritère terminent chacun une journée en environ 0,06 s. Ces mesures incluent le coût des diagnostics calculés séparément de l’appel d’inférence.

Les données suggèrent que certains réseaux apprennent à intervenir utilement, mais la fréquence et la valeur des interventions restent sensibles à l’observation, à la graine et au stade de l’apprentissage. Les données ne permettent pas d’isoler la contribution du crédit temporel, du bruit de valeur ou de l’échelle de récompense. Les deux tailles de flotte, trois dates, actions bornées, pénalités et tournées synthétiques limitent la portée des écarts. La phase n’établit ni une convergence générale ni une performance en conditions commerciales réelles.

## Vérifications, artefacts et reproduction

Le contrôle Gymnasium et celui de Stable-Baselines3 passent pour le continu et le compact. Les 113 tests usuels, dont quatre régressions ajoutées, les 35 tests d’audit et `pip check` réussissent. Le test numérique de la mise à jour rapproche la perte calculée par SB3 de trois cibles connues, dont une décision masquée et une fin réelle. Chaque checkpoint réseau et son tampon sont rechargés sur CPU. Les tables rechargées restituent leurs valeurs et visites. Trente rejeux témoins des 15 nouveaux modèles rechargés reproduisent les métriques et les actions sur une journée à trois véhicules et une à quatre. Les empreintes, les rôles disjoints des scénarios, les totaux d’interactions et les horloges communes ont été vérifiés. Vérification des relectures (historical artifact: `results/experiments/dqn_evaluation/cpu_reloads.csv`)

| Exécution | Temps mural | Pic RSS (KiB) |
| --- | ---: | ---: |
| Contrôle initial interrompu, 0 interaction | 13,82 s | 528 596 |
| Présélection et diagnostic | 73,95 s | 591 060 |
| Revue sur validation | 6,20 s | 414 328 |
| Apprentissage et validation à trois graines | 620,78 s | 622 968 |
| Rejeu final et analyse appariée | 461,31 s | 645 928 |
| **Expérience totale** | **1 176,06 s** | **645 928** |
| Notebook Jupyter | 4,78 s | 155 492 |
| 113 tests usuels | 27,40 s | 553 348 |
| 35 tests d’audit | 5,33 s | 148 104 |

Le premier contrôle a exigé par erreur un indicateur de ponctualité parfaite au lieu de s’appuyer sur le validateur physique de fin de journée. Il s’est arrêté avant apprentissage. Le contrôle corrigé mesure les retards comme résultats opérationnels. Les deux tentatives apparaissent dans le journal et sont comptées dans la durée totale. Le notebook utilise un noyau Jupyter local et contient deux figures exécutées. Les checkpoints conservent les poids, l’optimiseur et le rejeu. Retrouver exactement la trajectoire d’entraînement au milieu d’une journée demanderait aussi l’état interne du simulateur et des générateurs de la collecte.

Les sauvegardes de cette phase comprennent le code de diagnostic et d’expérience, la configuration, le notebook exécuté et environ 16 Mio de résultats. Les répertoires `screen/` et `fit/` contiennent les réseaux, tampons, tables, courbes, traces et profils. Le protocole gelé, les deux manifestes de données, les métriques finales et les écarts appariés se trouvent à la racine des résultats. Notebook (historical artifact: `notebooks/09_1_dqn_evaluation.ipynb`) · [Artefacts](../results/experiments/dqn_evaluation)

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-dqn.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
out=$(mktemp -d /tmp/smartfmcg-9b.XXXXXX)
/usr/bin/time -v python -m src.rl.dqn_evaluation screen --output "$out"
/usr/bin/time -v python -m src.rl.dqn_evaluation review --output "$out"
/usr/bin/time -v python -m src.rl.dqn_evaluation fit --output "$out"
/usr/bin/time -v python -m src.rl.dqn_evaluation final --test-per-day 6 --output "$out"
cp notebooks/09_1_dqn_evaluation.ipynb "$out/"
SMARTFMCG_DQN_OUTPUT="$out" jupyter execute "$out/09_1_dqn_evaluation.ipynb" --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
git log -1 --oneline --grep='^Validate bounded DQN training and evaluation$'
```

**GO recommandé pour la phase 10 après approbation explicite.** Prendre la règle événementielle comme référence de coût et le multicritère comme référence de service parmi les règles. Garder le DQN continu comme comparateur scientifique. Les prochains résultats pourront affiner l’effet du générateur et de la taille de flotte sur cette comparaison.
