# l2_norm_bwd: complete derivative and row-index pair

`Correct.grid_correctness` proves the independent Real derivative for the
complete repaired row grid. `FPEquiv.grid_equiv` compares two actual FP32
implementations as `original ≡[[]] candidate`, including both safe successful
executions, every active DX cell and both exterior frames.

The production source remains `l2_norm_bwd_repaired.py`. The separate
`l2_norm_bwd_index_commuted.py` ports the benchmark's existing index-commutation
candidate to the repaired signed64 source: `row * stride_x_row`,
`row * stride_dx_row` and `row * stride_dy_row` become `stride_*_row * row`.
The production wrapper and admission checks are unchanged and still select
the repaired source. No performance claim is made.

`Kernels.lean` directly displays both `triton { ... }` sources. Their three row
products are the only changed statements. Both retain independent
X/DY/DX row and column strides, signed64 row/column casts, both FP32 load casts,
zero padding and re-masking, both reductions, the runtime FP32 `eps` register,
the arithmetic expression and the masked FP32 store. `tl.param eps : tl.float32`
declares the external scalar type without emitting an assignment. `IO.invocation`
supplies its runtime value. There is no generated source macro, placeholder
epsilon constant, or statement deletion.

For `q = sum(x*x) + eps` and `dot = sum(dy*x)`, the independent Real target is
`dy*(1/sqrt(q)) - dot*(1/q)*(1/sqrt(q))*x`. The existing total Real sqrt/division
conventions are retained; interpreting it as a derivative requires `q > 0`.
`Proofs/Real.lean` connects the padded finite sums to this formula through
Mathlib. `Correct.lean`, `Contract.lean` and the production Python are unchanged.

`IO.lean` declares both inputs, independent strides, the DX window and the
complete runtime-epsilon row schedule and the actual original/candidate views
on exactly that interface. `Proofs/PairIO.lean` remains a compatibility import. The main pair
retains `N ≤ block`, output-address injectivity and the original backed windows.
Inputs may overlap and use zero or repeated strides; every floating payload,
including epsilon special values, remains admitted. Zero rows and zero-width
raw instances retain their existing meaning. There is no finite-value guard.

`Proofs/Rewrite.lean` proves the three signed integer products commute in the
actual pointer statements. The remaining statements are unchanged. It derives equality
of complete typed states/failures and transports actual-source trace safety.
`Proofs/ProgramPair.lean` connects every invocation and complete row schedule to
that local proof, reusing `IOGrid.grid_execution` to construct both successful
runs. All active DX cells and both complete exterior frames are conclusions;
no success or desired-output equation is a premise. The numerical rule set and
rewrite domain are empty: this transformation uses no floating arithmetic law.

`Misc/NativeCorrect.lean` retains the optional source-to-native graph theorem,
including the selected reduction trees, fusion choices, `sqrt.approx.ftz`,
`div.full` and final negation/FMA. `Misc/TensorBridge.lean` retains the checked
Tensor and physical-byte applications. Required formula, safety, execution and
pair helpers remain under `Proofs/`; neither primary entry depends on `Misc/`.
No new native FP32, host or hardware conformance requirement is introduced.

The completion evidence is under
`.veritile-cache/two-task-completion-20261021/l2_norm_bwd/`: targeted Lean builds,
main/auxiliary checks, source-preservation and exact three-change Python AST
checks. Historical `kernel-validation.json` records earlier single-source
validation; current pair delivery is `delivery.json` in that completion folder.
The separate `#print_fp_assumptions` diagnostic hit its heartbeat limit; its
command and precise unfinished status are recorded in that evidence folder.
The main pair and standard-base axiom checks passed. No GPU run or numerical
transformation experiment is required for this index rewrite. The separate shared runtime grid-product TODO remains unchanged.

The subsequent source-readability cleanup is recorded under
`.veritile-cache/l2-readable-kernels-20261021/`. Both full programs are checked
equal to their previously accepted definitions, including region-port metadata.
The old placeholder-only `Tile.epsilon_argument` helper has been removed with
the unused templates; the actual runtime parameter remains in `IO.invocation`.
