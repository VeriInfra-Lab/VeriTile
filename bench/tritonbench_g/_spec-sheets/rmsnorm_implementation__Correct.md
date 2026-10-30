# Spec sheet — `bench/tritonbench_g/rmsnorm_implementation/Correct.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_implementation/rmsnorm_implementation.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed programs realize the full-row RMSNorm formula with
successful flat execution, active bounds and the complete memory frame.
This is exact real mathematics, independent of the backend FP equivalence. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXO : X ≠ O) (hWO : W ≠ O) (hsok : 0 < sok) :
    VeriTile.Spec.Real
      (ExactIO (typedIO input weight output v X W O sxb sxm sxk sw sob som sok N B eps)
        (fun _ _ xs ws t j => rmsImplStreamSpec N B eps xs ws t j))
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hXO : X ≠ O`
- `hWO : W ≠ O`
- `hsok : 0 < sok`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `full_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete floating-profile programs satisfy actual trace safety,
successful finite flat execution, every active output of the independent full
row RMSNorm formula and the frame of all other raw cells. This is the exact-real
projection with nonnegative strides and zero undefined masked-load values;
storage casts erase to identity. It is not an IEEE arithmetic assertion. -/
```
</details>

**Statement:**
```lean
specification full_real_correct (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXO : X ≠ O) (hWO : W ≠ O) (hsok : 0 < sok) :
    VeriTile.Spec.Real
      (FullExactIO (typedIO input weight output v X W O sxb sxm sxk sw sob som sok N B eps)
        (fun _ _ xs ws t j => rmsImplStreamSpec N B eps xs ws t j))
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hXO : X ≠ O`
- `hWO : W ≠ O`
- `hsok : 0 < sok`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
