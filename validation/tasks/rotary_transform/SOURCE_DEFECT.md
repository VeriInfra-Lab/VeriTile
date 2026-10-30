# Admitted rotary widths exceed Triton's block-tensor limit

The current canonical source cannot successfully execute over its complete
admitted domain. Two nonempty, canonical contiguous FP32 inputs pass every
unchanged admission and formed-pointer check, then fail actual Triton frontend
compilation. This is independent of floating-point values, native primitive
semantics, FMA selection, and GPU numerical experiments.

The canonical source is `rotary_transform_repaired.py`, SHA256
`d59d9684bd9131a12e800b0a54859f06001de3f5b60f78b1e9f9eb625259eda8`.
Its JIT AST is still identical to the original source, SHA256
`920032ade0eb2da9a2cb26bbda36318ab1f74260be6b5b58f22427726a6399de`.

| Pairing | X shape | cosine/sine shape | BLOCK_K | First matrix shape |
|---|---|---|---:|---|
| Interleaved | `[1,1,1,262146]` | `[1,131073]` | 524288 | `[4,524288]` |
| Split | `[1,1,1,524290]` | `[1,262145]` | 1048576 | `[4,524288]` |

Both calls use fixed sequences, scalar offset zero, `conjugate=False`,
`inplace=False`, and one head. Input and cosine entries are 1; sine entries
are 0. All three allocations are separate, ordinary, zero-offset, canonical
FP32 tensors. No sequence metadata Tensor or coefficient copy is needed.
There is no nonrotary suffix. The independent mathematical target is 1 in
every output column: each pair gives `(1*1-1*0, 1*0+1*1)`.

The guard's padded data expressions are 2883602 and 5767186, respectively;
the padded coefficient expressions are 1703945 and 3407881. All are below
2^31. Shape, alignment, containment, storage separation, and unsigned64
formed-pointer checks pass. The wrapper selects grid `(1,1,1)`, BLOCK_M=4,
four warps, and the default compiler settings, including FP fusion.

At source line 56, the interleaved branch broadcasts `rm[:,None]` and
`rk[None,:]`. At line 39, the split branch broadcasts `rm[:,None]` and
`rk_half[None,:]`. Both require 2097152 elements. The source's row and column
masks occur later and cannot reduce the type's static shape. Even though only
one row is active, compilation fails with:

```text
ValueError('numel (2097152) exceeds triton maximum tensor numel (1048576)')
```

Triton 3.7.1 defines this frontend limit in `triton/_utils.py`,
`TRITON_MAX_TENSOR_NUMEL` and `validate_block_shape`. These witnesses are the
first even widths above this particular branch-dependent limit. The preceding
even widths meet this necessary frontend condition; no backend resource claim
is made for those large preceding widths.

The evidence has three independent parts:

- `Proofs/BlockSizeDefect.lean` verifies the complete unchanged `Checked.check`
  for concrete descriptors in both branches, independently of all payload
  bytes. It derives `Config.Valid` and `IndexABI`, proves the launch is nonempty,
  computes the exact selected block sizes, and disproves the claim that guard
  admission implies the frontend's necessary tensor-size bound. It also proves
  that the existing independent `realFormula` gives 1 for every column of the
  all-ones/zero-sine witness. This is an arithmetic/contract proof, not a Lean
  model of the external compiler.
- `source-defect-replay.md` uses the existing CPU-storage/CUDA facade to run the
  unchanged wrapper and both sets of guards, captures its actual launch
  arguments, and compiles the original JIT through Triton's actual argument
  binder. All four failing profiles (two pairing modes, SM80 and SM90) produce
  the expected frontend error. Four width-514 controls compile to PTX through
  that same path and default settings. `source-defect-evidence/report.json`
  contains descriptors, signatures, constexpr values, options and artifact
  hashes. Error traces and positive-control PTX are retained alongside it.
- The unchanged CPU regression suite still passes its 78 complete source/host
  paths and 16 rejections. Its existing small-width cases do not exercise the
  compiler's block-size restriction.

No GPU execution is claimed. CUDA availability is false in this environment;
the frontend rejection and positive controls are real offline CUDA compilation,
while Tensor admission is replayed with the existing CUDA facade over actual
CPU storage. The witness does not rely on an allocation or resource exhaustion.

Reproduce the compiler evidence from the repository root:

```sh
python - <<'PY'
from pathlib import Path
p = Path('validation/tasks/rotary_transform/source-defect-replay.md')
code = p.read_text().split('```python\n', 1)[1].split('\n```', 1)[0]
exec(compile(code, str(p), 'exec'))
PY
```

The kernel review therefore delivers a source-defect packet instead of a false
complete source-success theorem. Python, guards, shared core, Lake, and the
ledger were not changed. A future repair must address the source's block shape
or explicitly revise its admitted width domain before full kernel acceptance.
