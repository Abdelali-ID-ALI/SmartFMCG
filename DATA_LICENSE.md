# Licensing and attribution

| Material | Terms |
| --- | --- |
| Original project software | MIT applies to original code in `src/`, `dashboard/` and `tests/`, original notebook code cells, and original application configuration. [MIT license](LICENSE) |
| Original documentation and scientific material | CC BY 4.0 applies to the original prose, explanatory notebook text and eligible outputs, synthetic business tables, original scientific tables, figures, screenshots and frozen models, to the extent of rights held by their author. Credit Abdelali Id Ali and SmartFMCG, link to the license and indicate changes. Existing third-party notices remain applicable. [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) |
| OpenStreetMap-derived geographic inputs | ODbL 1.0 applies to the OpenStreetMap database and applicable derived databases. The exact routing files and mixed records listed in the reference manifest require separate cleared access. [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) · [Reference files](docs/reference_files.txt) |
| Third-party software and assets | Their upstream terms apply. This distribution uses official Python dependency packages and does not bundle their wheels. [Third-party notices](docs/third_party_notices.md) |

## Road data and models

**© OpenStreetMap contributors.** Static road distances, estimated travel times and route geometry used in the historical experiments were computed with the OSRM car profile through the **FOSSGIS e.V. routing service**. The project generated the orders, vehicle assumptions, traffic observations and disruptions. Its trained ML and RL artifacts were fitted with synthetic observations that used the archived OSRM reference. The model license covers the original model artifacts to the extent of the author's rights. The source data keep their own terms. [OpenStreetMap copyright and attribution](https://www.openstreetmap.org/copyright) · [FOSSGIS routing service](https://routing.openstreetmap.de/about.html) · [Report or fix a map error](https://www.openstreetmap.org/fixthemap) · [OSMF model guidance](https://osmfoundation.org/wiki/Attribution_Guidelines#Machine_learning_models)

The seven historical reference files are needed for the qualified interactive demonstration. The release includes their hashes. Obtain exact copies from a source authorized to provide them. Access to the underlying derivative database must be provided where required under the ODbL. [Artifact verification](docs/local_installation.md) · [Reference checksums](docs/release_artifacts.sha256)
