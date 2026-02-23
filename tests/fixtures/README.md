# Fixture inputs to provide

Place your concrete sample files here to turn synthetic tests into dataset-backed cases.

Suggested mapping:

- `nci_out/`: real `nci_<oname>.out` outputs (clustered promolecular first)
- `geom/`: matching `mol1` / `mol2` geometry files (`.xyz` preferred)
- `cp/`: matching `<filename>_CPs.xyz`
- `charges/`: optional `*_charges.dat` files for deterministic charge-model tests

Use anonymized or minimal examples where possible.
