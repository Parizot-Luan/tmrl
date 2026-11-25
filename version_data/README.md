# Version data

Documentation des versions du projet TMRL.

| Version | Run | Statut | Config | Archive données |
|---|---|---|---|---|
| V1 | `SAC_CNN_GRU_PROGRESS_v1` | terminée, conservée | `configs/v1.0_replay30min_mem200k_alpha005.json` | `~/TmrlData/versions/v1/` |
| V1.1 | `SAC_CNN_GRU_PROGRESS_v1_1` | abandonnée | `configs/v1.1.json` | `~/TmrlData/versions/v1.1/` |
| V2 | `SAC_CNN_GRU_PROGRESS_v2` | active | `configs/v2.0.json` | en cours |

## Fichiers

- `v1.md` : réseau, image, paramètres, résultats et restauration V1.
- `v1.1.md` : expérience abandonnée, changements de reward.
- `v2.md` : version active, crop 40 %/7 %, image 48×48, AMP et correction des replays.
- `v2.1.5-changements.md` : détail et justification des changements depuis `v2.1.4`.

## Principe

- Une version = un `RUN_NAME`.
- Chaque version garde ses poids, checkpoints, benchmarks et rewards.
- Les gros artefacts sont stockés dans `~/TmrlData/versions/`.
- Les explications et paramètres sont versionnés dans ce dossier.
