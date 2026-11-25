# Phase 10 — Benchmark scientifique final de SmartFMCG

## Objectif et hypothèses

Le benchmark mesure la contribution de chaque composant à une journée de livraison et de collecte FMCG. L’optimisation peut réduire le coût de la flotte et modifier les départs et l’ordre de visite. Une prévision peut mieux représenter le trafic tout en conduisant à un plan moins robuste. Les replans peuvent sauver une urgence, mais entraînent une coordination, un délai de diffusion et des changements de tournée. Ces trois hypothèses sont évaluées dans un même moteur, depuis une commande connue au dépôt jusqu’au retour des véhicules.

Un écart apparié s’écrit `Δᵢ = Jᵢ(A) − Jᵢ(B)` sur le scénario `i`. Exemple à la main avec deux plans qui ont les mêmes commandes et la même circulation. Si A coûte 1 580 MAD et B 1 600 MAD, Δ vaut −20 MAD, soit −1,25 %. Si leur OTIF vaut 90 % et 92 %, l’écart de service vaut −2 points. Ce cas représente un arbitrage. Une configuration est dominée sur le front perte–OTIF lorsqu’une autre a une perte au plus égale et un OTIF au moins aussi élevé, avec un gain strict sur au moins un axe. Ce front descriptif ne fixe ni un tarif pour la ponctualité ni une règle de choix hors de ces scénarios.

## Protocole gelé et contraintes communes

Le dépôt de départ était propre sur `9a214e3`. Le benchmark recharge le pipeline linéaire sur le ratio et reconstruit à l’identique les ratios historiques calculés sur les 34 jours d’entraînement de phase 5. Le fichier de modèle et le cache OSRM sont contrôlés par empreintes. La moyenne historique, son repli global, le modèle, l’heure de référence de 11 h 30 et la marge de 0,08870674 fois la durée OSRM sont figés avant le premier cas. Les checkpoints de B9 proviennent de la variante tabulaire A, qui minimise la perte moyenne de validation 9B parmi A, B et D. B10 utilise les trois réseaux continus sélectionnés en 9B, chacun à son checkpoint de validation. Ces six artefacts et le témoin compact sont utilisés en inférence CPU.

| ID | Plan initial | Politique pendant la journée |
| --- | --- | --- |
| B0 | Insertion gloutonne HF-VRPSPD-TW avec OSRM statique | Maintien |
| B1 | OR-Tools avec OSRM statique | Maintien |
| B2 | OR-Tools avec historique fixe | Maintien |
| B3 | OR-Tools avec ML fixe sans marge | Maintien |
| B4 | OR-Tools avec ML fixe et marge | Maintien |
| B5 | Plan B4 | Règle aux événements éligibles |
| B6 | Plan B4 | Règle périodique |
| B7 | Plan B4 | Règle à seuil |
| B8 | Plan B4 | Multicritère |
| B9 | Plan B4 | Q-Learning tabulaire A |
| B10 | Plan B4 | DQN continu |
| B11 | Plan B4 | Tirage uniforme parmi les actions publiques admissibles |
| B12 | Plan B4 | DQN compact, comparaison des représentations et algorithmes |

Le glouton de phase 4 rejoue les quatre dimensions de charge et les créneaux du problème complet. Le plus proche voisin de phase 3 optimise un CVRP homogène et son espace de solutions diffère du HF-VRPSPD-TW. B0 utilise donc le glouton validé de phase 4. OR-Tools dispose de 0,70 s par plan au standard, et B4 est calculé une fois puis réutilisé par ses politiques. Le glouton termine naturellement après sa construction. Le simulateur et le budget CP-SAT de dispatch de 9B restent la référence commune. Les comparaisons B1/B0, B2/B1, B3/B2, B4/B3 et B5–B11/B4 représentent respectivement l’optimisation, l’historique, le ML, la marge et l’adaptation. B12 et B9 partagent les mêmes classes d’observation d’origine et le même masque, à budget d’entraînement de 4 000 interactions par graine. Ce contraste compare la table au réseau avec leurs règles de mise à jour, d’exploration et de choix différentes. B10–B12 compare les deux observations sous le même DQN, avec sélection de checkpoint sur validation. B12 a été gelé avant la production des résultats principaux et rejoue exactement la même banque avec les plans B4 sauvegardés. Protocole principal (historical artifact: `results/experiments/final_benchmark/frozen_protocol.json`) · [Protocole du témoin compact](../results/experiments/final_benchmark/approximation_protocol.json)

Les 35 cas standard couvrent sept dates synthétiques inédites du 8 au 14 juin 2026 et cinq régimes par date. Les distributions de volume livré et repris, les fermetures de créneau et les flottes de cinq ou six véhicules varient selon un plan de tirage sans référence aux résultats des politiques. Les 30 sites et les routes OSRM restent fixes. Les événements varient en fréquence, en heure, en intensité de congestion, en incertitude de service et en quantité ou délai urgent. Leur calendrier part des fenêtres commerciales de référence et de tirages indépendants des tournées. Le régime normal n’introduit aucune urgence. Le trafic latent conserve les variations, pics et incidents synthétiques sur tous les régimes. Les dates et graines du développement et des essais de sensibilité sont distinctes. [Configuration](../configs/final_benchmark.yaml) · Manifeste des scénarios (historical artifact: `results/experiments/final_benchmark/scenarios.json`)

Le stock régulier total au dépôt et les réserves chargées sont identiques au sein de chaque scénario. La réserve contient le même nombre de caisses ambiantes et réfrigérées sur V06, dont l’activation est imposée par les deux commandes froides obligatoires. Les planificateurs déduisent cette réserve de sa capacité sur chaque dimension. Le simulateur utilise la capacité réelle et le chargement validé. L’affectation du stock régulier aux véhicules découle des tournées. La concentration de la réserve limite les options de secours et fait partie de la frontière de généralisation. Les kilogrammes, litres, caisses, positions palettes, départs, calendriers, distances et fenêtres sont partagés ou vérifiés selon leur rôle. B0–B3 conservent leur plan et le trajet réalisé est toujours tiré du même scénario. B4 à B12 prennent leurs décisions sous la même prévision de dispatch et observent les mêmes annonces dès leur révélation. L’éligibilité peut varier avec les engagements, les propositions en cours et l’histoire propre de la politique. Le calendrier des événements et des points de décision est contrôlé par empreinte pour chaque scénario.

## Métriques et incertitude

Le coût d’exploitation simulé additionne mise en service, distance et durée à bord, attente et service compris. Les coûts unitaires sont des hypothèses en MAD. Le coût avec omissions ajoute les pénalités des commandes facultatives non livrées. La perte commune de phase 8 ajoute retard, dépassement horaire, coordination et modifications de tournée. La somme de ces comptes est réconciliée avec le registre incrémental. Le temps de conduite est la somme des trajets du journal entre départ d’une jambe et arrivée, hors service et attente. L’OTIF compte les commandes livrées complètement avant fermeture. Le service urgent compte les livraisons, ponctuelles ou tardives, parmi les urgences apparues. Les cas sans urgence sont exclus de ce taux. Le taux de réalisation obligatoire mesure la livraison complète, indépendamment de la ponctualité.

Le replan appliqué compte une modification de tournée effectivement diffusée. L’instabilité compte les visites insérées ou dont le prédécesseur change, rapportées aux 30 commandes régulières. Les statuts de refus préventifs, les actions invalides et les violations physiques réalisées sont des événements distincts. La validation physique contrôle les contraintes de stock, de capacité, de chronologie, de compatibilité et d’engagement. Les retards observés sont conservés et pénalisés séparément. La mémoire par configuration est une mesure du pic RSS échantillonné du processus, avec les bibliothèques et modèles partagés. Le profil global rapporte le pic système du processus. Le temps de bout en bout attribue à chaque configuration le coût complet de son plan initial et, le cas échéant, la préparation commune des prévisions de la date.

On moyenne d’abord les graines au sein d’un cas, puis on mesure écart-type, médiane et quantiles 10 % et 90 % entre cas. Les comparaisons conservent les paires et calculent les différences absolues et les différences relatives quand le dénominateur est non nul. Les 4 000 tirages bootstrap reprennent des blocs de dates entiers et des graines d’apprentissage. Les deux agents partageant des identifiants utilisent les mêmes tirages de graines. Les cas d’un même jour partagent la géographie et le calendrier. Avec sept blocs et trois graines, les intervalles à 95 % décrivent l’incertitude de ce protocole, sans établir une généralisation commerciale. Un écart de coût faible peut compter moins en pratique qu’une différence de tournée, de délai ou d’effort de maintenance. Les tableaux de couverture recensent séparément tout statut de planification sans candidat. Les classements moyens utilisent les scénarios complets communs et les contrastes leurs paires disponibles.

## Résultats sur les journées communes

Les 12 configurations principales passent tous les contrôles de planification et de rejeu. Les moyennes donnent le même poids à chaque scénario et, pour B9 et B10, à chaque graine. Les appels CP-SAT de la table concernent la réoptimisation locale en cours de tournée. Les appels initiaux OR-Tools figurent dans les résultats individuels. [Tableaux complets](../results/experiments/final_benchmark/summary.csv) · [Figure de bout en bout](../results/experiments/final_benchmark/figures/01_end_to_end.pdf)

| ID | Perte (MAD) | Exploitation (MAD) | Avec omissions (MAD) | OTIF | OTIF régulier | Urgences servies | Retard (min) | Tentatives | Replans appliqués | Changements | CP-SAT en tournée |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B5 | 2 266,30 | 2 156,56 | 2 247,99 | 90,24 % | 93,52 % | 17,86 % | 7,94 | 3,00 | 0,57 | 1,26 | 0,00 |
| B8 | 2 266,76 | 2 157,17 | 2 246,31 | 90,89 % | 94,10 % | 21,43 % | 6,50 | 10,29 | 0,80 | 1,80 | 0,00 |
| B11 | 2 272,43 | 2 157,80 | 2 244,65 | 90,89 % | 94,00 % | 25,00 % | 7,06 | 22,37 | 0,80 | 1,94 | 7,11 |
| B7 | 2 277,12 | 2 153,00 | 2 260,43 | 89,99 % | 93,90 % | 7,14 % | 6,67 | 4,23 | 0,49 | 1,29 | 0,00 |
| B10 | 2 277,48 | 2 154,27 | 2 257,88 | 89,60 % | 93,33 % | 11,51 % | 9,86 | 1,48 | 0,25 | 0,55 | 0,00 |
| B6 | 2 277,80 | 2 159,91 | 2 249,05 | 90,80 % | 94,00 % | 21,43 % | 6,67 | 26,63 | 0,77 | 1,71 | 0,00 |
| B9 | 2 282,63 | 2 154,72 | 2 256,82 | 90,05 % | 93,75 % | 11,51 % | 7,28 | 20,18 | 0,50 | 1,30 | 6,21 |
| B4 | 2 284,34 | 2 152,49 | 2 264,49 | 88,90 % | 92,95 % | 0,00 % | 11,03 | 0,00 | 0,00 | 0,00 | 0,00 |
| B3 | 2 299,83 | 2 157,29 | 2 269,29 | 85,20 % | 89,05 % | 0,00 % | 16,97 | 0,00 | 0,00 | 0,00 | 0,00 |
| B2 | 2 315,42 | 2 173,34 | 2 285,34 | 86,10 % | 90,00 % | 0,00 % | 16,71 | 0,00 | 0,00 | 0,00 | 0,00 |
| B1 | 2 320,69 | 2 163,70 | 2 275,70 | 81,50 % | 85,24 % | 0,00 % | 24,99 | 0,00 | 0,00 | 0,00 | 0,00 |
| B0 | 2 500,88 | 2 334,49 | 2 475,06 | 89,49 % | 93,52 % | 0,00 % | 14,34 | 0,00 | 0,00 | 0,00 | 0,00 |

### Optimisation, prédiction et marge

B1 économise **180,19 MAD/jour** de perte par rapport au glouton B0, avec un intervalle descriptif de −206,21 à −150,53 MAD. Il parcourt 121,13 km contre 177,20 km et roule 206,19 min contre 294,60 min. Il perd toutefois **7,99 points d’OTIF** lors du rejeu des plans en trafic réalisé. Le glouton constitue un témoin opérationnel utile. La priorité aux fermetures et l’ordre des affectations y produisent des départs et des itinéraires différents. Le solveur optimise son objectif planifié. Un plan économe selon les temps OSRM statiques peut être plus sensible aux perturbations.

L’historique B2 gagne **4,60 points d’OTIF** face au statique B1, avec un intervalle de +3,86 à +5,43. Son écart de perte moyen de −5,27 MAD reste incertain. Le ML fixe B3 réduit la MAE sur les 35 cas, de 2,591 à 2,532 min en moyenne. Pour le ML horaire, elle descend de 1,394 à 1,261 min face à l’historique horaire. En présence d’incidents latents, le ML fixe atteint 8,58 min de MAE contre 8,55 pour l’historique fixe. Le ML fixe obtient une perte plus basse sur **17 cas sur 35** et un meilleur OTIF sur **10**. L’écart moyen B3–B2 vaut −15,59 MAD et −0,90 point d’OTIF, avec des intervalles couvrant zéro. La corrélation de Spearman entre l’écart de MAE fixe et l’écart d’OTIF est de 0,16. Une mesure moyenne sur tous les arcs n’accorde pas le même poids que le solveur aux créneaux, affectations et arcs critiques.

La marge figée B4 récupère **3,70 points d’OTIF** face à B3, avec un intervalle de +2,05 à +5,52 points. L’écart de perte de −15,49 MAD reste incertain sur ces sept dates. La marge aide à protéger les horaires dans cet essai et constitue la base commune de la seconde moitié du benchmark. [Contrastes appariés](../results/experiments/final_benchmark/paired_comparisons.csv) · [Erreur de prévision et résultat logistique](../results/experiments/final_benchmark/figures/08_accuracy_and_operations.pdf)

### Replanification et apprentissage

La règle événementielle B5 réduit la perte de B4 de **18,05 MAD/jour** avec un intervalle de −29,34 à −8,31 MAD et gagne **1,34 point d’OTIF**. Ce gain est une baisse simulée de la perte composite. Le coût d’exploitation augmente de 4,08 MAD/jour en moyenne. Le 8 juin en régime modéré, B4 coûte 2 207,74 MAD et B5 2 209,24 MAD. L’écart de 1,50 MAD vient des trois tentatives à 0,50 MAD sans changement de résultat. Les gains s’observent donc sur certaines journées.

B8 augmente l’OTIF de **0,65 point** face à B5, avec un intervalle descriptif de +0,19 à +1,21. Il augmente la perte moyenne de **0,46 MAD**, avec un intervalle de −3,89 à +3,12 MAD. Cette petite différence de coût s’accompagne de **10,29 tentatives** en moyenne contre **3,00**. Le périodique B6 améliore le service face à B4 avec 26,63 tentatives. L’aléatoire B11 obtient le meilleur taux urgent, 25,00 %, et un OTIF supérieur à B8 de 0,003 point en moyenne, avec 22,37 tentatives et 7,11 appels CP-SAT. Les trois points B5, B8 et B11 sont sur le front descriptif. L’écart d’OTIF entre B8 et B11 est négligeable pour le choix opérationnel. [Front perte–OTIF](../results/experiments/final_benchmark/figures/02_cost_service_pareto.pdf)

B10 gagne **6,86 MAD** et **0,70 point d’OTIF** face à B4, avec des intervalles de −14,35 à −1,34 MAD et de +0,27 à +1,25 point. Il demeure plus coûteux que B5 de **11,18 MAD**, avec un intervalle de +4,52 à +19,41 MAD. Il utilise 1,48 tentative par scénario et zéro CP-SAT en tournée. B9 utilise 20,18 tentatives et 6,21 appels CP-SAT, avec un OTIF supérieur de 0,45 point au continu en moyenne. L’écart de perte B10–B9 est de −5,15 MAD, avec un intervalle de −12,66 à +2,44 MAD. Les deux agents servent 11,51 % des urgences en moyenne et les trois graines du continu varient de 3,57 % à 16,67 %. L’inférence moyenne par décision prend 0,586 ms pour B10, 0,055 ms pour B9 et 0,003 ms pour B5. L’exécution de bout en bout prend respectivement 0,960 s, 1,305 s et 0,957 s, plan initial compris.

Le témoin compact B12 obtient **2 274,79 MAD**, **90,65 % d’OTIF** et **17,06 % d’urgences servies**. Face à la table utilisant les mêmes classes, son écart de perte est de −7,84 MAD, avec un intervalle de −25,14 à +7,16 MAD. L’effet de la famille d’approximation sur la perte reste incertain à ce budget. Le DQN continu perd 1,05 point d’OTIF face au compact, avec un intervalle de −1,95 à −0,14 point, et réduit les tentatives de 18,90 à 1,48. Le continu diminue l’effort de dispatch. La représentation apporte donc un changement de comportement dont l’intérêt dépend des métriques retenues. [Comparaison à information appariée](../results/experiments/final_benchmark/approximation_intervals.csv) · [Variabilité des graines](../results/experiments/final_benchmark/figures/06_learned_seed_variability.pdf)

## Robustesse et sensibilité

Le normal favorise B3 en perte moyenne. Le modéré favorise B8. B5 arrive premier en régime sévère et sous forte pression urgente. B7 et B5 sont proches avec des ressources limitées, à 2 381,61 et 2 381,83 MAD. Le meilleur coût moyen dépend donc du régime. Les différences entre régimes combinent demande, flotte, fenêtres, réserves et événements. La sensibilité traite ces facteurs individuellement sur trois autres dates. [Régimes](../results/experiments/final_benchmark/figures/04_disruption_regimes.pdf)

Les 45 variantes de sensibilité reprennent trois tirages modérés du 22 au 24 juin. Chacune change un facteur parmi la pénalité urgente (40 ou 120 MAD), la diffusion (5 ou 60 s), la réserve (3 ou 14 caisses par produit), le multiplicateur de congestion (1,15 ou 2,05), le surcroît de service nominal (4 ou 19 min), la fermeture des créneaux (−15 ou +15 min) et la disponibilité de la flotte (sans V01 ou sans V05). L’instance régulière et les tirages latents sont conservés autant que le facteur le permet. Les variations de réserve, fenêtres et flotte recalculent B4 avec la même méthode et le même budget. Les variations liées au dispatch ou à une réalisation cachée rejouent le plan B4 commun. Les politiques, leurs seuils et les poids restent gelés.

B5 minimise la perte dans les 15 conditions, contrôle inclus. Pour B5, retirer V01 augmente la perte de 228,39 MAD sur ces trois cas. Resserrer les fermetures de quinze minutes l’augmente de 21,68 MAD. Diminuer le stock de 9 à 3 caisses par produit ramène le service urgent de 66,67 % à 0 %. Avec 14 caisses, B9 atteint 88,89 % de service urgent contre 66,67 % pour B5, avec une perte moyenne supérieure de 14,18 MAD. Les hypothèses de réserve peuvent donc changer le classement de service. Les extrêmes de latence testés changent peu la perte et laissent le service urgent moyen inchangé sur ces trois dates. La conclusion est propre aux déclenchements et durées testés.

Sur les 49 urgences des 35 scénarios principaux, 14 ont un témoin de service effectivement réalisé par au moins une configuration principale parmi B0 à B11. Seize sont impossibles avec le stock initial et les compatibilités, compte tenu de l’absence de transfert ou de réapprovisionnement. Les 19 autres restent indéterminées. La rareté des occasions démontrées limite l’intérêt d’un agent capable de réagir, surtout sous la contrainte de servir régulièrement les 30 arrêts. Les résultats de sensibilité portent sur les conditions étudiées. Effets et rangs (historical artifact: `results/experiments/final_benchmark/sensitivity_ranks.csv`) · Diagnostic des urgences (historical artifact: `results/experiments/final_benchmark/urgent_diagnostics.csv`)

## Recommandation opérationnelle

- **Défaut pour la future démonstration, B5.** Il minimise la perte moyenne dans le benchmark principal et les 15 conditions de sensibilité parmi les cinq politiques évaluées. Son déclenchement aux annonces est explicable, sa moyenne de trois tentatives est faible et il ne lance aucun CP-SAT en tournée. Il combine un modèle de prévision figé, une marge documentée et une seule règle à maintenir.
- **Priorité au service, B8.** Il atteint 90,89 % d’OTIF total, 94,10 % d’OTIF régulier et 21,43 % d’urgences livrées. Il améliore le service de B5 avec plus de modifications. B11 sert 25 % des urgences et partage le même OTIF à la précision opérationnelle, avec davantage de calcul et de tentatives. B8 est la référence gouvernable lorsque la ponctualité prime.
- **Coût d’exploitation seul, B4.** Son exploitation moyenne de 2 152,49 MAD est la plus basse. Le calcul de perte inclut la valeur accordée aux urgences, aux retards et aux modifications.
- **Calcul minimal, B0.** Il prend 0,132 s par scénario et se déploie avec la seule matrice statique. Son coût moyen supérieur limite son intérêt comme défaut. B5 constitue le meilleur compromis observé parmi les options actives légères.
- **Agents appris.** B12 minimise la perte moyenne des trois agents appris rejoués. B10 sollicite le moins de replans et gagne face au plan robuste immobile. Les comparaisons avec B5 et B8 et la variabilité entre graines motivent leur emploi comme comparateurs scientifiques. La forte réserve relève le service de certains agents sur trois journées de sensibilité, avec une perte toujours supérieure à B5.

Les données soutiennent l’intérêt d’optimiser les tournées et de protéger les horaires, puis d’intervenir sélectivement. Elles montrent aussi qu’une baisse de l’erreur prédictive peut laisser le service stagner ou se dégrader. La réserve et les capacités d’insertion limitent la valeur des décisions adaptatives. Pour juger la signification pratique, le tableau de bord pourra montrer ensemble exploitation, perte, OTIF, service urgent, tentatives et instabilité. Les écarts de quelques centimes ne justifient pas à eux seuls une décision opérationnelle.

## Validité externe et limites

Les 30 sites reposent sur une carte OSM de Casablanca et une matrice statique OSRM dirigée. La géographie des arcs est réelle et les commandes, volumes, horaires, frais et réalisations du trafic sont synthétiques. Le profil automobile statique ne fait pas l’objet d’une validation poids lourd. La disponibilité de trafic en direct, les temps de chargement réels et la diversité géographique exigeraient des données supplémentaires. Les règles d’accès et de froid sont des règles du modèle. La perte de MAD exprime des coûts directement simulés et des pénalités hypothétiques. Une économie d’entreprise nécessiterait une calibration des tarifs, des pénalités et de la qualité de service.

Les tests portent sur 30 sites et cinq ou six véhicules disponibles, avec quatre à cinq véhicules utilisés en moyenne selon le plan. Ils stressent le générateur et la taille de flotte, avec une seule géographie et un mécanisme connu. Ils n’établissent aucun optimum global pour les tournées de la recherche bornée. Les agents voient des états agrégés et l’information reste partielle. Des interventions utiles sont rares, le stock est concentré, le budget d’apprentissage hérité reste limité et les graines changent les arbitrages. Les sept dates standard et les trois dates de sensibilité limitent les conclusions sur un changement de régime. Les essais sur les tarifs et la latence mesurent les hypothèses retenues, avec un protocole de variation séparée. Les interactions entre facteurs restent ouvertes.

## Validation, ressources et reproduction

Les 560 relectures principales, 405 relectures de sensibilité et 105 relectures du témoin compact terminent sur CPU. Aucun défaut physique, action invalide, repli de sécurité ou ordre obligatoire omis n’est enregistré dans ces relectures. Les sous-problèmes CP-SAT sans candidat sont recensés séparément. Les plans partagés, horloges, stocks, caractéristiques des scénarios, retours incrémentaux et correspondances d’empreintes ont été vérifiés. Les contrôles indépendants couvrent aussi les omissions, la fin de service tardive physiquement cohérente et la frontière de révélation des événements. Les 123 tests usuels, 35 tests d’audit et `pip check` réussissent.

Le moteur standard dure **333,30 s** (5 min 33 s) et atteint **628 744 KiB** de pic RSS. Le développement dure 71,73 s et le rejeu compact 37,91 s. L’agrégation des contrastes, des blocs et des graines prend 81,51 s. Les exports graphiques prennent environ 20 s par rendu. Les réexécutions du rendu et du notebook améliorent la lisibilité de la figure de Pareto. Le standard et les traitements utilisent les ressources CPU locales avec deux threads numériques. Le profil recherche est configuré pour un éventuel calcul ultérieur. [Profil des ressources](../results/experiments/final_benchmark/resource_profile.csv)

Les livrables comprennent les modules de scénario, évaluation, analyse et visualisation, la configuration, les plans et métriques bruts, les contrastes et rangs, huit figures en PDF vectoriel et PNG 300 dpi, et le notebook exécuté. Les fichiers des phases précédentes sont préservés. Notebook (historical artifact: `notebooks/10_final_benchmark.ipynb`) · [Résultats](../results/experiments/final_benchmark) · [Figures](../results/experiments/final_benchmark/figures)

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -r requirements-dqn.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
out=$(mktemp -d /tmp/smartfmcg-10.XXXXXX)
/usr/bin/time -v python -m src.benchmark.experiment --profile development --output "$out/development"
/usr/bin/time -v python -m src.benchmark.experiment --profile standard --output "$out/standard"
/usr/bin/time -v python -m src.benchmark.approximation --output "$out/standard"
/usr/bin/time -v python -m src.benchmark.analysis --output "$out/standard"
/usr/bin/time -v python -m src.benchmark.figures --output "$out/standard"
cp notebooks/10_final_benchmark.ipynb "$out/"
SMARTFMCG_BENCHMARK_OUTPUT="$out/standard" jupyter execute "$out/10_final_benchmark.ipynb" --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
python -m pip check
git log -1 --oneline --grep='^Validate the paired end-to-end scientific benchmark$'
```

La recherche OR-Tools est bornée en temps mural. Le temps par candidat et l’incumbent peuvent varier d’une machine à l’autre. Pour reproduire les décisions à plans initiaux identiques, ajouter `--replay-plans results/experiments/final_benchmark/plans.json` à la commande standard. Le moteur vérifie les empreintes des problèmes et valide les tournées sauvegardées. Les durées de planification restituées dans ce rejeu proviennent des mesures originales.

**GO recommandé pour la phase 11 après approbation.** Les composants ont été évalués dans le même protocole et les résultats sont utilisables pour une démonstration locale. Le défaut recommandé est B5, avec B8 pour illustrer le compromis de service et les agents appris dans la comparaison scientifique.
