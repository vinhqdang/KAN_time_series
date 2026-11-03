# Validation (seeds 5000..5004, 20 subgraphs): mean directed AUROC by config and resolution

| method | config | 6h | 12h | 24h |
|---|---|---|---|---|
| corr | {} | 0.651 | 0.647 | 0.650 |
| dynotears | {"lam": 0.001, "p": 1} | 0.534 | 0.494 | 0.470 |
| dynotears | {"lam": 0.01, "p": 1} | 0.620 | 0.594 | 0.548 |
| dynotears | {"lam": 0.1, "p": 1} | 0.660 | 0.642 | 0.638 |
| dynotears | {"lam": 0.3, "p": 1} | 0.597 | 0.589 | 0.607 |
| dynotears | {"lam": 1.0, "p": 1} | 0.500 | 0.500 | 0.500 |
| notears_svar2 | {"lam": 0.01, "p": 1} | 0.563 | 0.510 | 0.491 |
| notears_svar2 | {"lam": 0.1, "p": 1} | 0.589 | 0.579 | 0.561 |
| notears_svar2 | {"lam": 0.3, "p": 1} | 0.584 | 0.576 | 0.557 |
| notears_svar2 | {"lam": 1.0, "p": 1} | 0.584 | 0.576 | 0.557 |
| pcmci+ | {"alpha": 0.01, "p": 1} | 0.657 | 0.678 | 0.688 |
| pcmci+ | {"alpha": 0.05, "p": 1} | 0.648 | 0.672 | 0.661 |
| r2_sortnregress | {} | 0.631 | 0.644 | 0.648 |
| random | {} | 0.565 | 0.582 | 0.496 |
| var | {"p": 1} | 0.584 | 0.576 | 0.557 |
| var_sortnregress | {} | 0.544 | 0.491 | 0.520 |
| varlingam | {"p": 1, "prune": false} | 0.608 | 0.621 | 0.607 |
| varlingam | {"p": 1, "prune": true} | 0.603 | 0.638 | 0.635 |
