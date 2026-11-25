# CVRP classique

Le TSP ordonne les clients sur une tournée unique. Le VRP ajoute l'affectation des visites à plusieurs véhicules. Le CVRP fixe une capacité de chargement identique pour chaque véhicule dans cette première formulation.

Soit 0 le dépôt, C l'ensemble des clients, K celui des véhicules et A l'ensemble des arcs dirigés disponibles. Pour chaque client i, qᵢ est le poids livré. Pour chaque arc disponible (i,j), dᵢⱼ est la distance en mètres, arrondie à partir des valeurs OSRM. Q représente la capacité en kg par véhicule. Les variables binaires xᵢⱼₖ indiquent si le véhicule k emprunte l'arc (i,j).

- Minimiser la somme de dᵢⱼ xᵢⱼₖ sur les arcs et véhicules.
- Chaque client possède exactement un arc entrant parmi tous les véhicules.
- Chaque véhicule vérifie la conservation de flux chez les clients, avec au plus un départ du dépôt et autant de retours que de départs.
- Le poids affecté au véhicule est la somme de qᵢ multiplié par l'indicateur de visite de i. Cette somme ne dépasse pas Q.
- Les visites forment des tournées raccordées au dépôt. Dans une formulation binaire, on peut imposer que chaque sous-ensemble non vide de clients ait au moins un arc sortant vers son complément, dépôt compris. Dans RoutingModel, le solveur construit les chemins des départs aux retours.

Dans le code, les callbacks d'OR-Tools fournissent dᵢⱼ et qᵢ. La dimension Weight cumule la demande des clients déjà parcourus à partir de zéro. Pour une tournée de livraison pure, le cumul à l'arrivée au dépôt est égal au poids chargé au départ. Les arcs indisponibles sont retirés des domaines de successeurs. Le validateur indépendant recompte les visites, les poids et la distance dirigée de chaque arc, y compris les deux jambes du dépôt.

Exemple en mètres avec deux clients de 6 kg, deux véhicules de 10 kg et quatre arcs de dépôt D→A = 10, A→D = 40, D→B = 20 et B→D = 30. Chaque véhicule visite un client. Le coût total vaut 100 m. Avec une capacité de 12 kg, un véhicule peut visiter les deux et l'ordre devient déterminant. Si A→B = 5 et B→A = 25, D→A→B→D mesure 45 m et D→B→A→D mesure 85 m.

## Algorithmes comparés

- Le plus proche voisin choisit la prochaine visite accessible qui tient dans la capacité restante et permet un retour direct au dépôt. Il est rapide et déterministe. Une construction échouée fournit les clients non servis et distingue cet échec des certificats simples de capacité.
- La recherche locale part du plan glouton et évalue inversions 2-opt, échanges et déplacements. Elle recalcule la distance dirigée et les contraintes pour chaque candidat. Elle n'adopte qu'une amélioration faisable.
- OR-Tools combine PATH_CHEAPEST_ARC ou SAVINGS avec la descente gloutonne ou la recherche locale guidée. La méthode SAVINGS exploite l'économie estimée en reliant des visites auparavant desservies séparément. La recherche guidée pénalise temporairement des arcs pour explorer au-delà d'un minimum local. Le meilleur plan est comparé sur la distance originale.
- Un échec générique, un délai sans solution et un statut d'infaisabilité sont distingués. Un plan faisable reçoit une garantie d'optimalité uniquement si le statut natif l'indique.

## Expérience

Les 30 commandes sont imposées et la flotte est ramenée à six véhicules homogènes de 2 300 kg. La distance est la seule composante de l'objectif. Les créneaux, collectes, pénalités, restrictions d'accès, chaîne du froid et coûts en MAD relèvent de la phase suivante. Ces distances mesurent une instance de référence. Les temps observés incluent la construction et dépendent de la machine. La limite d'une seconde borne la recherche locale et celle du solveur, hors coût préalable de construction.
