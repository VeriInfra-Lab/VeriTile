# Mixed-output token softmax: selected floating routes

Seven new Lean modules add one public contract for each original token-softmax
kernel. A shared source takes independent input and output storage selectors;
each ranges over fp16, bf16 and fp32. All nine combinations are covered, including
six mixed pairs. All existing source and proof files remain unchanged.

Each public specification includes equality with the complete historical source
surface after dtype erasure, successful typed execution, exact active-lane values
from an independent initial-state formula, and preservation of every cell outside
the write footprint. The original masked load, source-to-fp32 conversion,
maximum/subtraction/exp/sum/division graph and destination conversion are retained.
Padding converts fp32 negative infinity into source storage before the load.
The theorem has no execution/result premise. Positive width and nonzero output
token stride are its explicit mathematical shape premises.

The homogeneous specialization is definitionally equal to the existing typed
kernel for the three selected storage types. Both original Python JIT functions
were extracted without importing or running their surrounding modules; their ASTs
are identical. Eighteen offline raw TTIR generations succeeded at sm90,
BLOCK_SIZE=1024 and four warps. The connected SSA audit checks the actual input
conversion, reduction/exp/division path, output conversion and masks. No native
backend compilation or GPU execution was performed.

The selected destination-dtype gap is closed. Whole-program acceptance stays zero:
the actual host block/warp choice and complete batch/head grid, other original
dtype/admission routes, and transport from the inherited total typed-cell model
to the original storage/layout/address domain remain separate. Source erasure and
frontend observations do not prove compiler or native backend correctness.
Numerical followups are descriptive and unvalidated; none are registered into the
production exact queue by this package.

`evidence/receipt.json` pins sources, public statements, scope, dependency objects,
frontend outputs and compilation history. The successful final build is bracketed
by identical dependency snapshots. Failed compiler/application attempts are kept.
The workspace contains symlinks to imported cached objects for module resolution;
those objects are dependency evidence and are not new source artifacts.

Compile from the repository working directory with the existing toolchain:

```sh
/tmp/veritile-lean-toolchain/bin/lake env python3 -B /absolute/package/verify.py <modules listed in modules.json>
```

The integration payload consists only of the seven `.lean` files in `modules.json`
and the frozen evidence. No live repository edits, existing-theorem replacements,
or original-program acceptance changes are part of this package.
