# Routage hétérogène avec livraison, collecte et créneaux

## Formulation

On utilise un dépôt 0, des clients i, des véhicules v et quatre dimensions k de charge (poids, volume, caisses, palettes). Les paramètres dᵢₖ et pᵢₖ indiquent la livraison et la collecte, Qᵥₖ la capacité et aᵢᵥ la compatibilité. La distance δᵢⱼ et le temps τᵢⱼ utilisent les arcs dirigés OSRM. Le service sᵢ doit commencer après l'ouverture eᵢ et finir avant la fermeture lᵢ. Les fenêtres du dépôt et du poste de travail bornent le départ et le retour.

Une variable binaire xᵢⱼᵥ sélectionne l'arc (i,j) pour le véhicule v et yᵢᵥ indique la visite. La sélection des arcs respecte la conservation du flux chez chaque client et relie les tournées au dépôt. Chaque commande obligatoire impose Σᵥ yᵢᵥ = 1. Pour une commande facultative, Σᵥ yᵢᵥ + zᵢ = 1, où zᵢ indique une omission. On impose yᵢᵥ ≤ aᵢᵥ pour les règles de froid et d'accès. Si un arc est choisi, l'heure de service suivante satisfait tⱼ ≥ tᵢ + sᵢ + τᵢⱼ. L'écart est du temps d'attente.

Pour une tournée active, l'objectif additionne son coût fixe, sa distance multipliée par le tarif kilométrique et sa durée écoulée multipliée par le tarif horaire. La durée inclut le trajet, le service et l'attente entre départ et retour. Le plan ajoute Σᵢ pénalitéᵢ × zᵢ. La pénalité de 125 MAD par commande facultative est une hypothèse configurable. Le carburant n'entre pas dans un poste supplémentaire à ce stade.

## Charge physique

Au départ, la livraison restante vaut Σᵢ dᵢₖ yᵢᵥ et la collecte cumulée vaut zéro. À chaque visite, le véhicule décharge, puis collecte. Après la visite, le restant diminue de dᵢₖ, la collecte augmente de pᵢₖ et la somme donne la charge à bord. Le restant reste non négatif puisque le départ contient exactement les livraisons de cette tournée. À l'instant intermédiaire, le déchargement ne peut qu'alléger le véhicule. Les charges au départ ainsi qu'avant et après chaque visite suffisent donc à contrôler les bornes sous cette convention. L'enchaînement D → A → B avec 9 kg au départ, 3 kg livrés puis 5 kg collectés chez A, donne 11 kg après A. Un véhicule de capacité 10 kg peut servir les mêmes clients en commençant par B si celui-ci libère les 6 kg restants.

## Traduction en OR-Tools

Pour chaque dimension, le cumul `Delivered` commence à zéro et augmente de la livraison à chaque départ de client. Le cumul `Onboard` utilise la variation collecte moins livraison et démarre avec une valeur libre. L'égalité `Onboard(start) = Delivered(end)` lie cette valeur à la somme des livraisons affectées. Les deux cumuls respectent les capacités par véhicule. La visite calcule la charge après service au cumul du successeur. Le validateur reconstruit séparément les livraisons restantes et retours, sans lire les dimensions internes.

La dimension `Time` combine service, trajet et attente. Son coût de portée est le tarif horaire. Les restrictions éliminent les couples véhicule-client et les arcs indisponibles. Les disjonctions ne concernent que les clients facultatifs. Les quantités internes sont en grammes, litres, caisses, palettes, mètres et secondes. Les coûts entiers utilisent 18 000 unités par centime, pour représenter exactement la conversion d'un tarif par km ou par heure sur ces unités. Le coût du plan est réconcilié avec l'objectif du solveur.

## Comparaison et limites

Le glouton insère les commandes obligatoires par date de fermeture, puis essaie les facultatives si l'insertion coûte au plus leur pénalité. Il rejoue la charge et l'horaire à chaque candidat. Un passage arrière borne la date de départ pour minimiser l'attente tout en respectant les fermetures. Un échec d'insertion décrit les obstacles rencontrés. OR-Tools explore davantage de combinaisons sous budget et sépare l'infaisabilité, le délai et l'absence générique de solution. Une omission facultative du solveur reçoit un diagnostic d'éligibilité. Une cause globale plus précise demanderait une expérience contrefactuelle.

Le modèle suppose un dépôt, une tournée par véhicule actif, une commande par client, un service indivisible et la disponibilité des marchandises au dépôt. Les durées sont statiques et la convention décharge-avant-collecte est fixe. Les palettes retournées sont absentes de l'instance synthétique, bien que leur dimension accepte des collectes dans les tests. Les résultats sont issus d'une instance et d'un budget.


## Équations et conventions de remédiation

Pour chaque véhicule v, on distingue les nœuds 0⁺ et 0⁻ comme deux copies du même dépôt. Si uᵥ indique qu’il est actif, Σⱼ x₀₊ⱼᵥ = Σᵢ xᵢ₀₋ᵥ = uᵥ. Pour chaque client i, Σⱼ xⱼᵢᵥ = Σⱼ xᵢⱼᵥ = yᵢᵥ. Dans RoutingModel, chaque ensemble de clients sélectionnés doit appartenir au chemin reliant le départ au retour. Pour tout xᵢⱼᵥ = 1, tⱼᵥ ≥ tᵢᵥ + sᵢ + τᵢⱼ. Les horloges de départ et retour sont bornées par l’intersection dépôt/poste, et eᵢ ≤ tᵢᵥ ≤ lᵢ − sᵢ quand yᵢᵥ = 1. Le coût du véhicule actif comprend Fᵥ + cᵥ Σᵢⱼ δᵢⱼ xᵢⱼᵥ + hᵥ(t₀₋ᵥ − t₀₊ᵥ), dans des unités cohérentes, auquel s’ajoutent les pénalités d’omission au niveau du plan. En fixant toutes les collectes à zéro on retrouve des livraisons avec créneaux. Fixer également une flotte homogène et supprimer les restrictions donne la variante VRPTW homogène.

Les durées de trajet et de service sont arrondies à la seconde supérieure. Une durée strictement positive ne devient pas nulle. Pour les minutes OSRM obtenues par conversion de secondes, une tolérance d’un ULP traite les artefacts binaires à la frontière d’une seconde entière. Une durée source de 2,4 s entraîne 3 s planifiées et un service de 1,9 min entraîne 114 s. Le coût est calculé sur cet horaire conservateur. Toute charge initiale déclarée est comparée à la somme des livraisons. OR-Tools expose sa valeur embarquée au départ et le validateur la contrôle.

Sur graphe incomplet, pour au plus cinq clients et quatre véhicules, le solveur réserve une partie de son budget à une construction de secours. Si OR-Tools termine sans plan, cette construction teste des affectations complètes puis les ordres de visite et renvoie le premier plan faisable vérifié, si le budget le permet. Elle ne certifie pas son optimalité. Un échec générique reste distinct du délai et d’une infaisabilité prouvée. Le stock réel par produit sera un contrat différent lors de la simulation dynamique.
