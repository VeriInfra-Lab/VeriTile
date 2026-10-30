# Spec sheet — `bench/tritonbench_g/index_select_cat/Correct.lean`

**Python source:** `bench/tritonbench_g/index_select_cat/index_select_cat.py`

## Public theorem: `source_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete source variants implement the independent gather cell formula
from the original signed-index and source input windows. Successful flat
execution and the full frame are included. Duplicate source rows, overlapping
output strides and input/output aliasing require no injectivity premise. -/
```
</details>

**Statement:**
```lean
specification source_real_correct (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    VeriTile.Spec.Real (FullGatherIO input output index v out idx inp n c s0 s1 BI BC)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
