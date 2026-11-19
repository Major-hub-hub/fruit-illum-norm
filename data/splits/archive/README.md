# Historical split

`split_seed42_ungrouped.csv` preserves the manifest before exact-content grouping.
It contains 3,200 files, with seven byte-identical pairs. Two pairs cross splits:

- `FreshOrange (74).jpg` (train) and `FreshOrange (87).jpg` (test).
- `FreshOrange (75).jpg` (test) and `FreshOrange (88).jpg` (validation).

Original CSV SHA-256:
`734761953d6c81cb2828e0f3cf96c339a4dd9624ca9431af77499aa401ae6d7a`.

This is a provenance record. The active manifest is
`data/splits/split_seed42.csv`, which keeps exact-content groups in one split.
