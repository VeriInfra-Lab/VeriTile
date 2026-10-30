# Spec sheet — `bench/tritonbench_g/rbe_triton_transform/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rbe_triton_transform/rbe_triton_transform.py`

## Public theorem: `rbe_triton_backend_correctness`

<details><summary>docstring</summary>

```
/-- Complete accepted pair surface: actual fp16 storage, fp32 computation scopes,
both masked outputs, initial-state reference and frame. The numerical algebra
is still an explicit parameter: native transcendental/mixed-cast binding remains
open; this is not a closed software bit theorem. -/
```
</details>

**Statement:**
```lean
specification rbe_triton_backend_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : FPTyped.Parameters) (s : State α) (hw : FPTyped.WritesApart p s) :
    ∃ final, FP.Structural.exec M (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPTyped.value M p s)
```

**Assumptions / layout contracts:**
- `hw : FPTyped.WritesApart p s`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
