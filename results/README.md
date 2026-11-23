# Results tables

- `final_queue_v3_v3b.csv`: the final queue (0.7 kg box, offsets +-0.15 m, no privileged inputs, sensor noise 0.05, validation-checkpoint trainer). 30 evaluation episodes per subgraph, three subgraphs. Successes out of 30 per subgraph; pooled out of 90. Source: `LOG.md` sections 50-53.
- `earlier_queues.csv`: v1 (first trainer, noisy checkpoints) and v2 (validation trainer, 0.5 kg box). Not comparable with v3.
- `pairwise_tests.csv`: Fisher's exact tests on pooled episodes. Descriptive only: episodes within a run are not independent and there are three subgraphs.

Repeated open-loop evaluations on identical box offsets are not identical (10/20 in one run, 5/20 in another), so differences of a few episodes are within noise.
