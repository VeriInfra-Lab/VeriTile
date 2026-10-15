# Matrix transpose finite acceptance proposal

The new `after/bench/tritonbench_g/matrix_transpose/FullScope.lean` compiles and
proves the complete original wrapper outcome relation under the explicit
primitive boundary in `bench/original_program_acceptance.md`. It composes the
existing genuine full fp16 interpreter proof; it does not introduce a successful
execution or final-value oracle. This producer package remains pending independent
root review and repository integration. It adds zero accepted tasks by itself.

The exact original JIT AST generated frontend IR for all 231 configurations
`m = 2^a, d = 2^b, a + b <= 20`. Every one of the other 210 pairs whose two
individual aranges are legal fails at the first matrix pointer broadcast's
1,048,576-element tensor cap. Fourteen additional invalid/boundary calls were
checked. These are Triton 3.7.1 frontend checks, not native backend or GPU runs.
The Lean theorem proves that this finite domain is exactly the complement of
nine first-rejection classes over all natural dimensions, so the 231 cases are
exhaustive, not selected examples. Both original test sizes are included.

The exact wrapper AST was independently replayed with CUDA requests recorded and
Torch metadata allocations redirected to the meta device: all 231 shape/stride,
fp16, contiguous allocation, grid, parameter, and exact returned-Out identity
checks passed. Another 96 combinations check ordinary primitive-error propagation
and source-order precedence; eight negative/non-shape boundary calls are retained.
A boolean in the second dimension can pass randn and fail at zeros after shape
reversal. The package does not assert a universal first-call error for arbitrary
Python objects. Canonical natural dimensions are the supported shape API; Python
conversion, negative/non-shape arguments, and allocator/driver/runtime failure
remain ordinary trusted API behavior, not additional successful input routes.

The public outcome theorem includes the actual returned tensor metadata and all
existing successful-route claims: every sampled fp16 word transposed, equal shape
products, input preservation, and a frame for every cell outside the allocations.
It propagates errors at randn, zeros, launch setup, source frontend validation,
and post-frontend backend/runtime. The existing original interpreter must still
execute successfully; the theorem proves this fact. All four primitive error
flags are universally quantified and are not theorem hypotheses of success.
There are no numerical callbacks or deferred numerical assumptions.

`admitted_offset_and_byte_bounds` proves every element offset < 2^20 and both
bytes of each fp16 element < 2^21. Nonnegative intermediate stride products are
bounded by those offsets. Thus the original int32 stride/offset operations have
no signed overflow on any admitted wrapper shape. Fresh allocations, normal
same-format load/store, and compiler/device implementation remain explicit
trusted primitives, as permitted by the finite standard. Native functional
correctness is still not established.

Evidence and proposals:

- `evidence/fullscope-final.log`: successful Lean elaboration, six standard-axiom
  checks, and the independent result's non-circularity check.
- `evidence/frontend-receipt.json`, `frontend-success.json`,
  `frontend-rejections.json`: complete finite frontend matrix and exact source
  hashes; representative raw IR files preserve load/store/transpose evidence.
- `evidence/trusted-frontend-primitives.json`: installed arange and block-shape
  validator sources, including the exact tensor-size constant.
- `evidence/wrapper-binding.json`: source-call, metadata, return-identity, and
  error-order checks, with trust limits made explicit.
- `scope-cases.json`: 231 exact successful configurations, nine complete natural
  frontend-rejection partitions, and four primitive-error partitions.
- `catalog-supplement-task.proposed.json`: retains the existing historical
  declaration and appends the new public execution contract/source view.
- `scope-row.proposed.json`: schema2 coverage, with status still `pending`, and
  the old initial/partial-progress evidence retained. Root must supply review,
  regenerate current bindings/catalog, and run the unchanged acceptance checker.
- `integration-inputs.json`: exact source/public statement hashes and current
  canonical catalog/supplement references.

The test helper is separately inventoried and qualifies as `test_harness`: it is
non-JIT, is not called by either production function, and only calls wrapper then
clones output. There are no production Python control-flow branches to omit.

Root integration must add the new source (already covered by the existing Lake
glob), merge the proposed supplement task, supply a substantive independent
review with exact source bindings, regenerate the catalog, bind the final schema2
row/review, and run ordinary source/build/trust/comparator gates. Historical
receipts and old declarations must stay frozen. The scope gate must not be
weakened. No repository or Git changes were made by this producer.
