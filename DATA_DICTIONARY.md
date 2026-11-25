# Phase-one data dictionary

Business records and travel-time realizations are synthetic. Casablanca neighbourhood names and illustrative coordinates provide geographic context. The default static traffic reference uses the qualified OSRM road matrix. Its input and projected points are recorded separately. CSV timestamps are local Casablanca wall-clock times, in ISO 8601 format without an offset. Identifiers are strings. Dimensions and costs refer to the generated instance, without external calibration.

## depots.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| depot_id | ID | Fixed D01. |
| latitude, longitude | decimal degrees | Fixed illustrative depot near Aïn Sebaâ. |
| opening_time, closing_time | HH:MM | Fixed 07:00–18:00. |

## customers.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| customer_id | ID | Sequential C01–C30. |
| customer_type | category | Grocery, supermarket, restaurant, cafe or wholesaler. One of each per zone, with rotated ordering. |
| latitude, longitude | decimal degrees | Zone centre plus independent uniform offsets ±0.003° on each axis, except C26–C30, fixed at distinct vertices of the archived Aïn Diab road geometry. |
| geographical_zone | category | Aïn Sebaâ, Sidi Bernoussi, Roches Noires, Maarif, Bourgogne or Aïn Diab. |
| time_window_start, time_window_end | HH:MM | Three hours, beginning at 08:00 through 12:00 by slot. The witness requires service completion before closing. |
| service_duration | min | 8 min setup plus 1 min per started group of six crates. |
| priority | category | High for supermarkets, standard otherwise. |
| vehicle_access_restrictions | category | `no_heavy` for grocery, cafe and restaurant outside Bourgogne. `any` otherwise. This is a scenario rule. |

## vehicles.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| vehicle_id | ID | Fixed V01–V06. |
| vehicle_type | category | Two large vans, two medium trucks, one heavy truck and one refrigerated truck. |
| maximum_weight | kg | Fixed payload by vehicle, 2,300–6,000 kg. |
| maximum_volume | m³ | Fixed usable volume, 10–30 m³. |
| maximum_crates, maximum_pallets | counts | Fixed stowage limits of 195–500 crates and 4–12 pallet positions across vehicles. |
| fixed_operating_cost | MAD/active vehicle-day | Fixed illustrative activation, 160–390. |
| cost_per_km | MAD/km | Fixed illustrative variable tariff, 2.2–5.0. |
| cost_per_hour | MAD/h | Fixed illustrative operating tariff, 45–75. |
| fuel_consumption | L/100 km | Fixed illustrative rate, 10–25. |
| refrigerated | boolean | True only on V06. |
| shift_start, shift_end | HH:MM | Fixed 07:30–17:30. |

## orders.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| order_id, customer_id | ID | Sequential O01–O30 and customer foreign key. One order per customer in phase one. |
| product_type | category | Chilled juice for the restaurant and supermarket in Aïn Diab, ambient beverage elsewhere. |
| quantity | bottles | 12 per delivered crate. |
| delivery_weight, delivery_volume | kg, m³ | 12 kg and 0.032 m³ per delivered crate, including its packaging. |
| delivery_crates | count | Rounded normal by category with centres 16, 46, 22, 10, 65 in customer-category order above and standard deviation 14% of the centre. Clipped to 5 through 118% of the centre rounded down. Larger outlets order larger batches. |
| delivery_pallets | count | One position per started group of 40 crates for supermarkets and wholesalers, zero otherwise. Payload and volume use the crate equivalents and omit independent pallet tare. |
| pickup_crates | count | Binomial per delivery crate with probability 0.56 for cafes/restaurants and 0.28 elsewhere, to represent denser return loops on premises. This is a volume proxy. |
| pickup_weight, pickup_volume | kg, m³ | 2 kg and 0.020 m³ per empty crate. No return pallets. |
| refrigerated_required | boolean | True for chilled juice only. |
| priority | category | Inherited from the customer. |
| mandatory | boolean | True except for cafes, which are optional orders in this scenario. |
| order_date | date | Shared configured service day. |

## traffic_observations.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| timestamp | local date-time | Uniform draw over the 56 previous days and minutes in hours 06–21. Ranked chronologically. Repeated observations are allowed. |
| origin_id, destination_id | ID | Uniform sample among ordered depot/customer arcs with positive distance and duration in the selected reference. |
| road_distance | km | Qualified directed OSRM distance by default, rounded to four decimal places. Proxy mode uses 1.28 × Haversine distance and is reserved for bootstrap or test data. |
| static_travel_time | min | Qualified directed OSRM duration by default, rounded to four decimal places. Proxy mode uses 22 km/h for local and 35 km/h for other trips. |
| departure_hour, day_of_week | int | Derived from the timestamp. Monday = 0. |
| road_characteristics | category | Synthetic zone-relation flag. Local for same-zone pairs, arterial otherwise, including depot legs. It does not identify measured OSM road classes. |
| congestion_indicators | category | Incident with independent probability 3.5%, normal otherwise. Contemporaneous simulator outcome reserved for descriptive analysis and evaluation because the generator makes it unavailable at departure. |
| observed_travel_time | min | Static estimate multiplied by a factor with base 1.04. Add 0.46 on weekday peaks or 0.15 on weekend peaks (07–09 and 16–18), 0.12 when a central zone is touched, 0.08 for local trips, 0.65 for incidents and normal noise with SD 0.12. Clip factor below at 0.8. |

## events.csv

| Field | Unit/type | Generation and meaning |
| --- | --- | --- |
| event_id | ID | Sequential E01–E05 by event type. |
| timestamp | local date-time | Fixed event slots on service day, sorted chronologically. |
| event_type | category | Congestion, service delay, urgent order, cancellation or extra pickup. |
| target_id | ID/category | Uniform customer draw for customer events and that customer's zone for congestion. |
| severity, unit | numeric/category | Congestion multiplier uniform 1.3–2.0. Service delay uniform integer 8–25 min. Urgent request uniform 5–15 crates. Cancellation fixed at one order. Extra collection uniform 2–12 crates. |

Events describe inputs to future simulations. The nominal feasibility replay checks the original orders, including optional orders, using one fixed vehicle per zone, unload before pickup, and an illustrative 22 km/h proxy on every arc. This verifies an explicit feasible assignment under synthetic travel assumptions.

The configured OSRM reference is shared by the traffic generator and routing experiments. The generator verifies distance and duration against the directed reference at CSV precision. Raw static durations stay in minutes in the NPZ and CSV. The FMCG optimizer uses conservative whole seconds for feasibility. Simulated observed travel times are never routing input in phases 0–4.

## Scénarios dynamiques de phase 7

Les événements des comparaisons dynamiques sont générés par `src/simulation/experiment.py` à partir des dates, graines et régimes de `configs/dynamic_simulation.yaml`. Les tirages, les conditions et les horaires sont synthétiques. Le CSV de la phase 1 reste l’illustration de la génération initiale.

| Champ | Unité ou type | Définition |
| --- | --- | --- |
| day, seed, regime | date ISO, entier, catégorie | Clé d’appariement. Les politiques partagent la date, la graine, le régime et leurs réalisations. |
| at_s, end_s, extra_s | secondes | Heure depuis minuit de révélation et de fin annoncée, ou durée de service supplémentaire. |
| nodes, factor | indices, ratio | Sites de la zone touchée et multiplicateur des trajets démarrés pendant l’alerte. |
| stop.window, service_s | secondes | Créneau depuis minuit et durée nominale d’une commande urgente. Le site est un nœud routier déjà connu. |
| stop.delivery, starting_reserves, holdings | grammes, litres, caisses, positions palettes | Vecteurs physiques. Une caisse urgente contient 12 kg et occupe 32 L. L’état embarqué contient quatre compartiments, à savoir livraisons affectées, réserve ambiante, réserve réfrigérée et emballages. |
| planning_wall_s, modeled_latency_s, dispatch_hold_s | secondes, secondes, secondes-véhicules | Temps mural de préparation et validation, latence modélisée par tentative, attente cumulée des départs après service. `predeparture_hold_s` couvre le départ initial. |
| route_changes | compte | Insertions et changements du prédécesseur d’une visite du suffixe. |
| operating_cost_mad, realized_cost_mad | MAD | Coût d’exploitation et coût d’exploitation augmenté des omissions, respectivement. |
| otif, regular_otif, urgent_otif | proportion | Fin de service complète dans le créneau, sur toutes les commandes, les régulières ou les urgentes. |

## Apprentissage tabulaire de phase 8

| Champ | Unité ou type | Définition |
| --- | --- | --- |
| split, case, training_seed | catégorie, ID de scénario, entier | Partition figée, provenance de scénario et graine indépendante d’apprentissage. Les identifiants servent à l’évaluation et à la traçabilité. |
| state, visits, q_mad | tuple discret, entier, MAD | Observation agrégée à quatre composantes, nombre de mises à jour de la paire et valeur estimée. |
| objective_mad, episode_return | MAD | Perte journalière et son opposé non actualisé. Incluent exploitation, omissions, retards, dépassements, instabilité et coordination. |
| solver_calls, solver_no_incumbent | compte | Appels CP-SAT locaux et appels sans solution admissible retournée. Les statuts distinguent infaisabilité du sous-modèle et limite de recherche. |
| requested, executed, fallback_reason | catégories | Action demandée, action appliquée et motif d’un repli ou d’une résolution sans candidat. |
| actionable_decisions, unseen_decisions | compte | Décisions où le calcul est possible, et parmi elles celles dont la table évaluée n’a jamais rencontré l’observation. |

## Raffinement de phase 8.1

| Champ | Unité ou type | Définition |
| --- | --- | --- |
| minimum_best_slack_s, mean_best_slack_s | secondes | Minimum et moyenne, par urgence dotée d’un stock compatible, des meilleures marges directes prévues de fin de service. |
| mean_direct_arrival_slack_s | secondes | Moyenne de la marge fermeture moins arrivée pour les véhicules dont le service direct a la meilleure marge. |
| best_insertion_slack_s, best_gain_mad | secondes, MAD | Marge de l’insertion au meilleur gain et économie prévisionnelle nette. |
| probe_candidates, probe_wall_s, probe_exhaustions | compte, secondes murales, compte | Positions testées, temps cumulé à calculer les indicateurs et décisions dont la borne coupe la recherche. |
| advanced_omissions, total_omission_advance_s | compte, secondes cumulées | Omissions reconnues irréversibles avant la fin et temps gagné sur le règlement terminal par commande. |
| common_return, training_return | MAD | Retour de la comptabilité canonique et retour employé pour la mise à jour tabulaire. |

## Benchmark de phase 10

| Champ | Unité ou type | Définition |
| --- | --- | --- |
| block, variant, regime | catégories | Bloc principal ou sensibilité, niveau du facteur et régime opérationnel. La sensibilité répète le même tirage au sein d’un bloc de date. |
| scenario_sha256, traffic_sha256, plan_sha256, decision_clock_sha256 | empreintes | Identifient respectivement les entrées communes, le trafic latent, le plan validé et les heures et types des décisions. |
| planned_cost_mad, planning_status, dispatched | MAD, statut, booléen | Coût du plan selon son estimateur public, statut du calcul initial et réussite du passage à l’exécution. |
| realized_travel_min, vehicle_duration_min | min | Conduite réelle cumulée entre départ de chaque jambe et arrivée, et durées cumulées en service des véhicules. |
| urgent_service_rate, mandatory_completion_rate | proportions | Part des urgences livrées et part des commandes obligatoires livrées. Le taux urgent est manquant pour une journée sans urgence. |
| route_instability, route_changes, replans_applied | ratio, comptes | Prédécesseurs changés et insertions par commande régulière, nombre total de changements et nombre de diffusions modifiant la tournée. |
| initial_planning_wall_s, forecast_wall_s, planning_wall_s, inference_wall_s, end_to_end_wall_s | s murales | Construction du plan, prévision, recherche et validation des replans, calcul de l’action et temps total imputé à la configuration. |
| physical_rss_sample_peak_mib | Mio | Pic RSS échantillonné pendant le rejeu, bibliothèques et modèles compris. Le pic système global est donné par le profil d’exécution. |
| physical_violations, invalid_action_frequency, safety_fallback_frequency | compte, proportions | Violations réalisées du journal physique, demandes interdites et replis de sécurité rapportés aux points de décision. |

## Phase 10.1 — Géographie et profils de calcul

Les plans et registres de simulation conservent les unités des phases 4, 7 et 10. `S0031` et suivants sont des identifiants de sites synthétiques. Les coordonnées sont en degrés WGS84. Les positions OSRM ajoutées sont des sommets d’une trajectoire OSM archivée et espacés d’au moins 45 m entre points demandés. Le fichier de résultat `matrix.json` contient l’ordre dépôt–clients et la provenance du fournisseur. Le profil Haversine utilise une perturbation déterministe autour des sites de référence et une vitesse uniforme de 30 km/h.

| Champ | Unité et définition |
| --- | --- |
| `first_incumbent_s`, `planning_wall_s` | Secondes murales jusqu’au premier plan détecté et pour le calcul total. En dispatch, `planning_wall_s` cumule construction et validation des replans. |
| `modeled_latency_s`, `replan_p95_ms` | Secondes simulées pendant lesquelles un plan se diffuse, et 95e percentile par journée du calcul observé en millisecondes. |
| `process_peak_mib`, `matrix_array_bytes` | Pic RSS du processus en Mio (2²⁰ octets) et taille des deux tableaux NumPy de distances et durées stockés dans le cache en octets non compressés. |
| `prediction_rows` | Nombre d’arcs positifs et joignables multiplié par 24 tranches horaires. |
| `candidate_limit_hits`, `search_items` | Nombre de tentatives atteignant le plafond heuristique et nombre de candidats réellement examinés par décision. |
| `ood_box_eligible` | Décisions éligibles avec au moins une valeur DQN en dehors du minimum–maximum observé pour une composante dans le tampon de la graine correspondante. |
| `unseen_table_eligible`, `unknown_action_eligible` | Décisions éligibles dont l’état est absent de la table entraînée ou sans action disponible visitée. |
| `paired_loss_mad`, `paired_otif_pp` | Écarts vis-à-vis de B4 sur le scénario correspondant, en MAD et en points de pourcentage. |

## Phase 11 — Indicateurs de démonstration

| Champ affiché | Signification |
| --- | --- |
| Operational loss | Perte objective en MAD issue du registre Gymnasium. Inclut le coût d'exploitation et les pénalités configurées de service, d'omission, de changement et de tentative. |
| Loss savings vs B4 | Perte B4 moins perte de la politique, en MAD, pour le même scénario et plan initial. |
| Initial load utilization | Moyenne des fractions charge initiale en grammes sur capacité en grammes, sur les véhicules utilisés. |
| Known unfinished orders | Nombre de commandes régulières ou urgentes déjà annoncées, hors commandes terminées à l'heure du curseur. |
| Operational time / wall time | Horloge simulée depuis minuit, affichée en HH:MM:SS, ou durée de calcul mesurée en secondes respectivement. |
