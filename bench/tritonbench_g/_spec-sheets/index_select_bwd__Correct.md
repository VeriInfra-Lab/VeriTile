# Spec sheet — `bench/tritonbench_g/index_select_bwd/Correct.lean`

**Python source:** `bench/tritonbench_g/index_select_bwd/index_select_bwd.py`

## Public theorem: `source_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete source variants independently implement the last-active-write
scatter formula from the original signed-index and gradient input windows.
Successful flat execution and every untouched memory cell are included;
duplicate destinations and input/output aliasing require no injectivity premise. -/
```
</details>

**Statement:**
```lean
specification source_real_correct (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    VeriTile.Spec.Real (FullScatterIO input output index v out idx inp rows n c s0 s1 BI BC)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
