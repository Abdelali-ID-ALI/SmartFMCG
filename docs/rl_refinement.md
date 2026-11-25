# Phase 8.1 — Observation, crédit temporel et ablations

## Point de départ

La phase 8 a validé le moteur, le contrat Gymnasium et la réconciliation des coûts. Sur ses 36 cas finaux, le Q-Learning replanifiait 25,87 fois par jour et servait 12,04 % des urgences, contre 8,31 tentatives et 25 % d’urgences pour le multicritère. Les tables avaient déjà rencontré presque tous les états exploitables du test. Ce constat montre que manquer un état entièrement inédit expliquait peu de décisions. Il ne démontre pas que des états voisins ont les mêmes décisions utiles. La phase 8.1 isole l’observation et le signal d’apprentissage, puis vérifie ces effets sur de nouveaux tirages. Les cas et modèles de phase 8 restent des références historiques. [Résultats historiques](q_learning.md)

## Observation et coût d’information

L’agent accède aux annonces reçues, aux engagements, à la charge et aux prévisions utilisées par les règles de dispatch. Une sonde teste au plus 32 positions d’insertion pour les urgences en attente, à séquence régulière fixée et avec le délai de diffusion. Elle alterne entre demandes et propage les durées, le stock, les quatre dimensions de capacité, les fenêtres et le retour au dépôt. La position retenue pour calculer la marge est celle de meilleur gain prévisionnel. Une absence de solution dans cette sonde ne prouve pas l’infaisabilité, car un réordonnancement ou une position non examinée peut débloquer un cas. Chaque insertion est examinée séparément. La sonde ne certifie pas que plusieurs urgences peuvent être servies ensemble.

Pour chaque urgence, on retient la meilleure marge de fin de service direct pour un véhicule compatible disposant du produit. Le vecteur conserve le minimum et la moyenne de ces marges par commande, ainsi que l’écart moyen entre arrivée directe et fermeture. Les comptes de véhicules compatibles et approvisionnés, les réserves libres distinctes par produit, la charge de travail modifiable, le risque de retard et les temps depuis la dernière tentative et depuis la dernière application complètent cette description. La disponibilité du stock déduit les urgences déjà affectées. La prévision du bénéfice d’insertion combine transport, retards, dépassement, stabilité et omission évitée, puis retranche la coordination et une estimation de l’attente. Elle n’utilise que l’information connue au point de décision. Le rapport distingue le coût mural de la sonde et l’attente modélisée associée au lancement réel d’un replan.

L’observation continue est un vecteur `float32` de 22 composantes dans `[-1, 1]`, avec schéma nommé. Les grandeurs utilisent des normalisations fixes et les comptes indiquent quand une marge est indéfinie. Le vecteur est accessible dans l’adaptateur Gymnasium. Pour la table, on conserve le risque à trois classes, la charge à deux classes et le calcul récent à deux classes. Le contexte passe de cinq à sept catégories en distinguant l’absence de réserve, l’absence d’insertion trouvée, l’insertion fragile et l’insertion à marge supérieure à cinq minutes et gain prévu positif pour les demandes couvertes. La taille déclarée passe de 60 à 84 états, ou de 49 à 73 après prise en compte de l’état canonique indisponible. Les seuils sont fixés a priori. Le code mesure le support empirique, les paires visitées et les décisions hors support, ainsi que les préférences divergentes observées dans une même cellule.

La sonde reçoit le même budget pour tous les passages, y compris les politiques classiques et le témoin A. Les règles classiques restent gelées, avec leurs déclencheurs de phase 8. La sonde ne résout aucun CP-SAT, ne révèle ni les événements ni le trafic caché, et ne transmet aucune route au moteur. Le coût est évalué séparément afin de permettre une comparaison des latences d’inférence compatibles avec les mêmes informations publiques.

## Récompense et exemple à la main

Soit `c_t` l’accroissement de la perte opérationnelle déjà validée. Les incréments de transport et de temps, les retards, les dépassements, la coordination à 0,50 MAD par appel lancé et la modification du prédécesseur restent les mêmes. La récompense originale vaut `−c_t`. Une livraison urgente est irréversiblement omise si elle est connue, encore non affectée, non livrée et si l’heure courante augmentée de son service nominal dépasse la fermeture. Cette preuve ne dépend ni de la circulation future ni d’un échec de recherche. Une livraison déjà affectée reste exclue du règlement anticipé, même lorsqu’elle finit en retard. Le code tient un registre par commande. Au dernier pas, il retranche de la pénalité terminale d’apprentissage le montant déjà reconnu. Le registre canonique d’évaluation est inchangé.

Exemple. À 11 h 58, une commande non affectée demande 180 s de service et ferme à midi. Même avec un trajet nul, elle se terminerait à 12 h 01. La variante anticipée débite 80 MAD à la première décision qui établit ce constat. À la fin, le coût commun de 80 MAD est toujours présent et la correction d’apprentissage annule le second débit. Avec `γ = 1` la somme journalière est inchangée. Avec une actualisation inférieure à un, avancer un coût modifie sa valeur actualisée. C’est pourquoi l’ablation principale fixe `γ = 1`.

Deux autres retours sont testés séparément sur validation. Le premier attribue le coût kilométrique au départ d’une jambe engagée, puis retire cette anticipation à l’arrivée pour éviter un double débit. Le second ajoute `d_t Φ(o_{t+1}) − Φ(o_t)`, avec `d_t = γ^(Δt/900)`. Le potentiel est proportionnel à la pénalité des urgences connues non livrées, avec la moitié du poids pour une urgence déjà affectée. Il vaut zéro à la terminaison naturelle et conserve sa valeur à la troncature. Les produits des facteurs d’actualisation télescopent la somme des corrections en `−Φ(o_0)` pour un épisode complet. À état de départ fixé et facteur identique dans la cible de Bellman, l’ordre des retours des trajectoires reste inchangé. Cette identité n’implique pas que l’apprentissage tabulaire sur une observation partielle soit stable. Elle n’assimile pas non plus la reconnaissance anticipée sous actualisation à une transformation à potentiel. [Ng, Harada et Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) · [Sutton, Precup et Singh](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/Sutton-Precup-Singh-AIJ99.pdf)

Les signaux de coût déplacés sont réservés à l’entraînement. L’évaluation recalcule la perte de phase 8 et contrôle sa réconciliation pour chaque journée. Les contraintes opérationnelles et les masques restent appliqués par le moteur. Une prime apparente d’affectation dans le potentiel est compensée par les termes ultérieurs. Le fait d’expirer sans affectation ne déclenche aucun gain net. La fréquence de calcul, les replis et les changements sont suivis séparément pour détecter un comportement qui chercherait à éviter le service ou à multiplier les tentatives. L’étude vérifie aussi directement la sensibilité au facteur temporel et à une division de la récompense par 100. Une multiplication positive devrait préserver les classements à initialisation nulle et trajectoire identique, aux arrondis numériques près.

## Protocole apparié

| Variante | Observation | Retour d’apprentissage |
| --- | --- | --- |
| A | Phase 8 | Phase 8 |
| B | Raffinée | Phase 8 |
| C | Phase 8 | Omission reconnue plus tôt |
| D | Raffinée | Omission reconnue plus tôt |

L’ablation préspécifie 72 épisodes par graine, trois graines, la même séquence de tirages par graine et deux checkpoints à 36 et 72 épisodes. Le premier checkpoint est un point de la même trajectoire d’apprentissage, avec un niveau d’exploration plus élevé qu’en fin de parcours. L’effet d’une durée plus longue est donc évalué sous ce calendrier d’exploration, avec contrôle des autres facteurs. Les valeurs sont initialisées à zéro. Le pas reste 0,12 et l’exploration décroît de 0,90 à 0,08. Les comparaisons d’autres signaux sur validation reçoivent chacune 24 épisodes et une même graine. Elles servent au diagnostic sans sélectionner a posteriori une cinquième variante pour le test.

Les partitions standard comprennent 48 cas d’entraînement, 12 cas de validation et 45 cas finaux, séparés par jour au sein de la phase et par graines des expériences antérieures. Le test est instancié après l’apprentissage et les diagnostics. Les dates de référence et le réseau routier proviennent toujours de la bibliothèque synthétique déjà validée. Les nouveaux tirages varient la composition de flotte, les volumes réguliers, les réserves, la congestion, le délai de service et la présence éventuelle d’une seconde urgence. Aucune demande régulière ne disparaît du plan à cause de cette variation. Les fenêtres, capacités et règles de compatibilité sont conservées. Les intervalles appariés combinent d’abord les graines par cas puis rééchantillonnent les cinq dates entières. Cinq blocs restent une information limitée, notamment pour établir une différence faible. Les effets principaux sont les moyennes des deux contrastes conditionnels. L’interaction vaut `D − B − C + A`.

Pour mesurer l’aliasing indépendamment de la sonde, on rejoue sur validation les actions possibles à un même point, pendant une trajectoire multicritère. Dans chaque branche contrefactuelle, le reste de la journée conserve le plan. On relève si une action alternative aurait réduit la perte réalisée comparativement au maintien. On compare l’hétérogénéité de cette étiquette dans les cellules initiales et raffinées et l’erreur de majorité hors scénario. Ce diagnostic utilise la réalisation après coup, uniquement pour mesurer. Il ne fournit ni données d’entraînement ni étiquette disponible au moment de la décision. Il porte sur une déviation unique et une continuation fixe.

## Diagnostics sur validation

Sur 47 décisions avec une urgence en attente et une intervention disponible, trois présentent un bénéfice réalisé pour au moins une des actions testées avec continuation fixe. L’ancien regroupement contient une cellule mixte de trois décisions. Le raffinement sépare ces décisions dans cet échantillon. L’erreur par majorité évaluée en excluant le scénario concerné reste de 1 sur 47 pour les deux encodages. Le support empirique des trajectoires multicritères de validation passe de 25 à 26 états. La mesure confirme que l’agrégation fusionnait certaines décisions de valeur différente. Trois cas positifs sont insuffisants pour estimer précisément le pouvoir prédictif de la nouvelle observation.

| Signal d’apprentissage | Perte sur validation (MAD) | Service des urgences | Tentatives par scénario |
| --- | ---: | ---: | ---: |
| Original | 1 402,82 | 25,00 % | 26,00 |
| Omission anticipée | 1 403,09 | 25,00 % | 24,92 |
| Potentiel | 1 407,83 | 16,67 % | 22,25 |
| Distance au départ | 1 399,84 | 16,67 % | 22,00 |
| Actualisation par 15 min avec γ = 0,995 | 1 402,82 | 25,00 % | 26,00 |
| Récompense divisée par 100 | 1 402,82 | 25,00 % | 26,00 |

La reconnaissance anticipée a déplacé les débits de 24 omissions dans le diagnostic d’entraînement de la variante correspondante. L’avance moyenne est de 6,16 h sur le règlement terminal. Les contrôles vérifient que le total reste inchangé. Dans le diagnostic de mise à l’échelle, les compteurs de visites et les choix gloutons ont coïncidé. L’actualisation a également donné les mêmes métriques de validation sur cet essai. La distance au départ réduit légèrement la perte dans ce petit échantillon, avec un service urgent inférieur. Les conclusions sur le potentiel, l’actualisation et la distance restent limitées à 24 épisodes et une graine.

Les tables finales de A et C couvrent 34 à 36 états. B et D en couvrent 43 à 46, sur un espace déclaré plus grand. Entre les checkpoints 36 et 72, les choix changent dans 50 à 65 % des états exploitables communs selon la graine et la variante. Les effets du budget de 72 épisodes restent distincts de toute preuve de convergence. Le pas est constant, l’exploration décroît et les classes agrégées peuvent mélanger des transitions différentes.

## Résultats sur 45 scénarios finaux

Les 45 cas réunissent 62 demandes urgentes et de 14 à 20 clients réguliers par jour. Les taux sont les moyennes des taux par journée. Les variantes tabulaires moyennent trois graines. La perte commune additionne le coût d’exploitation, les omissions, le retard, le dépassement, la stabilité et la coordination. Mesures par scénario (historical artifact: `results/experiments/q_learning_refinement/test_cases.csv`)

| Politique | OTIF total | OTIF régulier | Urgences servies | Exploitation (MAD) | Coût avec omissions (MAD) | Retard cumulé (min) | Tentatives | Replans appliqués | Changements | Appels CP-SAT | Perte (MAD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Statique | 81,86 % | 88,31 % | 0,00 % | 1 286,27 | 1 396,49 | 17,51 | 0,00 | 0,00 | 0,00 | 0,00 | 1 428,02 |
| À chaque événement | 85,38 % | 90,26 % | 22,22 % | 1 292,26 | 1 375,82 | 13,10 | 3,38 | 0,69 | 1,49 | 0,00 | 1 404,07 |
| Périodique | 85,33 % | 90,59 % | 17,78 % | 1 288,99 | 1 377,88 | 13,46 | 24,62 | 0,64 | 1,44 | 0,00 | 1 417,29 |
| À seuil | 84,97 % | 90,84 % | 10,00 % | 1 288,44 | 1 386,22 | 12,49 | 4,16 | 0,58 | 1,29 | 0,00 | 1 413,35 |
| Multicritère | 86,06 % | 90,98 % | 22,22 % | 1 292,15 | 1 375,71 | 12,28 | 8,07 | 0,87 | 1,87 | 0,00 | 1 405,58 |
| Aléatoire | 85,46 % | 90,98 % | 15,56 % | 1 290,51 | 1 382,96 | 12,23 | 20,44 | 0,71 | 1,58 | 6,27 | 1 418,35 |
| A | 85,62 % | 90,94 % | 17,41 % | 1 290,95 | 1 380,43 | 12,33 | 20,73 | 0,77 | 1,69 | 2,76 | 1 416,36 |
| B | 85,07 % | 91,07 % | 9,26 % | 1 288,85 | 1 388,40 | 12,13 | 21,50 | 0,65 | 1,47 | 12,53 | 1 423,92 |
| C | 85,20 % | 90,86 % | 13,33 % | 1 289,47 | 1 384,28 | 12,65 | 18,27 | 0,63 | 1,43 | 10,27 | 1 419,04 |
| D | 84,77 % | 90,54 % | 11,85 % | 1 288,70 | 1 385,29 | 12,79 | 23,65 | 0,56 | 1,27 | 12,25 | 1 422,68 |

L’alerte à chaque événement obtient la perte la plus faible. Le multicritère obtient l’OTIF total le plus élevé avec le même service urgent. Le témoin A obtient la meilleure perte des variantes apprises tout en engageant 20,73 tentatives par jour. Toutes les politiques servent les commandes obligatoires. Aucun repli de sécurité n’apparaît dans les scénarios évalués. Les traces consignent les recherches sans amélioration et les appels CP-SAT.

Par rapport à A, B accroît la perte de **7,56 MAD** avec un intervalle bootstrap à 95 % de **+5,07 à +9,89 MAD**. C réduit les tentatives de **2,47** (−2,81 à −2,19), mais son OTIF perd **0,42 point** (−0,72 à −0,12) et ses appels CP-SAT passent de **2,76 à 10,27**. Son écart de perte est de **+2,68 MAD** (−0,69 à +6,03). D perd **0,85 point d’OTIF** (−1,41 à −0,50) et coûte **+6,31 MAD** (+2,97 à +9,79). En moyenne sur les deux niveaux de récompense, l’observation raffinée ajoute **5,60 MAD** (+3,77 à +7,54). L’effet moyen de la récompense anticipée est de **+0,72 MAD** (−1,62 à +3,32). L’interaction réduit de **3,92 MAD** (−8,09 à −0,40) le surcoût qu’impliquerait une addition des deux effets conditionnels de B et C. Ces intervalles sont descriptifs pour cinq blocs de dates et les trois tables évaluées par variante. Contrastes appariés (historical artifact: `results/experiments/q_learning_refinement/paired_intervals.csv`)

Entre 36 et 72 épisodes, la perte moyenne de validation baisse de 4,64 MAD pour A, 1,02 pour B et 3,57 pour D. Elle augmente de 3,13 MAD pour C. Les quatre variantes font davantage de tentatives à 72 épisodes. Sur test, les cas décisionnels inédits représentent 0,16 % pour A et C, 0,45 % pour B et 0,64 % pour D. Les tables originales couvrent 114 à 125 paires sur 240 déclarées et les raffinées 139 à 152 sur 336. La couverture absolue augmente et la densité baisse. Cette combinaison et les changements de politique entre checkpoints expliquent les limites d’une conclusion sur la convergence. Les retours C et D ont avancé 446 omissions d’en moyenne 6,20 h pendant l’entraînement. Leur perte journalière non actualisée reste identique au registre commun pour chaque passage.

## Ressources et préparation à la phase 9

| Exécution | Temps mural | Pic RSS (KiB) |
| --- | ---: | ---: |
| Développement | 35,98 s | 240 636 |
| Standard | 555,25 s | 315 520 |
| Notebook | 6,14 s | 267 140 |
| Tests usuels | 11,23 s | 243 628 |
| Régressions d’audit | 3,56 s | 147 972 |

En standard, les 864 épisodes d’ablation prennent en moyenne **0,236 s** avec **5,91 appels CP-SAT**. La planification occupe environ 84 % du temps d’épisode d’apprentissage. La sonde et son encodage coûtent **0,080 ms** par décision en moyenne et **0,374 ms** lorsqu’une urgence attend. Elle examine en moyenne 3,71 positions dans ces situations et atteint sa borne sur zéro décision finale. La durée standard dépasse l’estimation initiale de six à huit minutes. Les appels CP-SAT des politiques finales, notamment B et D, augmentent le temps de rejeu par rapport au pilote. La limite de calcul est principalement CPU et dépend de la distribution des actions. Le profil recherche est paramétré à 144 épisodes par graine et quatre véhicules. Il reste hors du budget exécuté.

Le checker Gymnasium, **101 tests usuels**, dont neuf en phase 8.1, **35 régressions d’audit** et `pip check` réussissent. Les 24 checkpoints JSON chargent et 12 relectures sur les tables finales restituent les métriques opérationnelles. Le notebook exécuté présente les courbes, les contrastes, les volumes d’intervention et les contrôles de récompense. Les artefacts de phase 8 sont préservés.

**GO pour préparer un pilote DQN borné, après approbation.** L’observation continue est disponible, les fenêtres et actions sont contrôlées, et la réconciliation de la récompense est vérifiée. Les résultats tabulaires montrent un effet défavorable de la discrétisation testée à ce budget. Ils laissent ouverte l’utilité des composantes continues et de la reconnaissance anticipée. Tester cette utilité requiert des graines et cas finaux renouvelés, des références classiques gelées, le contrôle de faisabilité pendant l’exploration et la mesure des appels solveur avant d’augmenter le budget. La phase 9 reste soumise à approbation explicite.

## Reproduction

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -m pip check
/usr/bin/time -v python -m src.rl.refinement_experiment --profile development --output /tmp/smartfmcg-rl-refinement-dev
/usr/bin/time -v python -m src.rl.refinement_experiment --profile standard
jupyter execute notebooks/08_1_rl_refinement.ipynb --inplace
python -m pytest -q
python -m pytest -q tests/audit_phases_0_4.py
```

[Configuration](../configs/rl_refinement.yaml) · Notebook exécuté (historical artifact: `notebooks/08_1_rl_refinement.ipynb`) · Tables, données et journaux (historical artifact: `results/experiments/q_learning_refinement`)
