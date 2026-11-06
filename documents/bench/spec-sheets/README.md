# Historical specification cards

The 347 cards and their index are retained in the
[fixed source snapshot](https://github.com/VeriInfra-Lab/VeriTile/tree/b1aaccc3ef23290fca80fe544fddb0844c036fd8/bench/tritonbench_g/_spec-sheets).
They describe earlier source revisions. The [historical contract review](../../../validation/archive/contracts.json) preserves
the conditions from the earlier review.

Generate cards from the current selected public declarations with:

```sh
python3 scripts/spec_sheet.py
```

The default output is `Logs/spec-sheets/`, which Git ignores. Use `--out DIR`
to choose another directory, or give explicit Lean files to print their cards.
Cards reproduce statements and local definitions; lexical flags do not certify
specification independence, proof correctness or native execution. The Lean
trust gates and official comparator remain separate checks.
