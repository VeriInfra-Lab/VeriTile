# vector_addition dtype rejection: proposed bounded structural package

The next package should prove a finite **admission model**, with an independently
checked correspondence audit of frozen Python frontend branches. It must not call
the existing exception observations a formal proof of Python/compiler execution.
The original-program completion increment remains 0.

## Two dimensions, with the actual first failure preserved

The scope is the existing 17 canonical scalar signatures, ordered operand pairs,
CUDA sm90, BLOCK_SIZE=1024 and the frozen Triton 3.7.1 frontend options. Let I be
the 9 integer/bool signatures, F the 4 fp8 signatures, and U the 2 fnuz signatures.

| Classification | Count | Meaning |
| --- | ---: | --- |
| Standalone promotion succeeds | 217 | A promotion result exists; this says nothing about signature admission. |
| Standalone promotion rejects | 72 | Exactly I×F and F×I. |
| First frontend stage: unsupported fnuz signature | 64 | Any pair containing U: 17²−15². |
| First frontend stage: addition promotion rejection | 36 | Remaining I×(F\U) and (F\U)×I. |
| Passes these selected dtype gates | 189 | This is not successful native lowering or GPU execution. |

Of the 64 signature rejections, 28 have successful standalone promotion and 36
have failed standalone promotion. The latter 36 never reach `x + y` on the
audited frontend path. All 289 existing observations match this cross partition.
Do not relabel the 28 rejected cases as additional value-execution proofs.

## Frozen source route to audit

1. Original `add` first calls `torch.empty_like(x)`, then the CUDA assertion,
   obtains `output.numel()`, and invokes the kernel with x/y/output signatures.
   It has no dtype, equal-shape, length, stride or same-device guard.
2. `triton._utils.type_canonicalisation_dict` maps fnuz to `fp8e4b8` and
   `fp8e5b16`; `str_to_ty` preserves those element types under pointer signatures.
3. `ast_to_ttir` constructs an `ASTFunction` in original argument order.
   `CodeGenerator.visit_FunctionDef` calls `prototype.serialize` **before**
   `visit_compound_statement(node.body)`. Serialization flattens argument types;
   `pointer_type.to_ir` calls the element dtype's `to_ir`.
4. `dtype.to_ir` raises `ValueError` when an fp8 name is outside
   `builder.options.supported_fp8_dtypes`. The frozen default sm90 set is
   `(fp8e4b15, fp8e4nv, fp8e5)`. Neither fnuz name belongs to it. Explicit option
   overrides or another backend/architecture require a different instance.
5. For admitted signatures, the original two loads produce tensor operands of
   their scalar element dtypes. Semantic `add` calls `binary_op_type_checking_impl`,
   which calls `computation_type_impl` with both scalar flags false and
   `div_or_mod=false`. Its final noninteger branch rejects fp8/integer mixtures.
   These are source typechecking errors, not numeric overflow/device errors.

The binding manifest includes full-file SHA256 values, exact text and normalized
AST hashes for 17 relevant functions, the unchanged kernel extraction AST check,
the original promotion/frontend evidence hashes, and the complete 289 cross rows.
It also retains full observed error text separately from the model predictions.

## Minimal implementation

* Reuse `DTypeScope.DType`, its explicit `promote`, `storeElement`, and exhaustive
  17-element list. Add a concrete sm90 configuration and an outcome type:
  `signatureUnsupported(first_argument, dtype)`, `additionTypeMismatch`, or
  `passesDTypeGates(promoted_dtype, output_store_dtype)`.
* Evaluate signature gates first, in x/y/output/n_elements order. Output's dtype
  equals x, so it cannot be the first unsupported fnuz argument after x passes.
  Evaluate the tensor-add promotion rule only after signature admission.
* Prove in Lean the exact rejection predicates, first-stage precedence, the
  disjoint/exhaustive 189/64/36 partition, the 217/72 promotion partition, and
  the 28/36 cross split of the 64 fnuz signature failures. Prove all results over
  explicit functions; no premise may assume the desired exception or acceptance.
* Name public declarations as **admission-model** or **stage-model** results.
  They should not assert that `torch.add`/the original `add`/the complete JIT
  pipeline throws an exception. No numeric callbacks are needed for this model.
* Independently freeze and audit the real canonicalization, signature recursion,
  guard condition, supported set, visit order, tensor operand status and
  promotion branch. An AST-shape guard must fail if those control dependencies
  change. Recheck all 289 existing evidence rows; record this as observation
  agreement, not a proof premise. No new GPU run is necessary for this package.
* Register all 289 rows with separate fields for standalone promotion, predicted
  first model stage, frozen source audit, observed frontend result, existing
  value contract, and unproved source/native refinement. Keep 189 value-admitted
  rows and 100 rejection-model rows separate, including the overlapping 36.

## Boundaries that must remain explicit

The minimal package is a kernel-checked finite model plus a source audit. Its
Python/AST-to-model correspondence is not itself a formal Python semantics proof.
That structural bridge cannot be postponed under a numerical-assumption label.
Host allocation/device assertions, original module-level test execution, JIT
cache/launch paths, unsupported or noncanonical Torch dtypes, exact exception
wrapping, target option overrides, lowering/runtime errors, debug sanitizers,
physical memory, and GPU execution remain separate obligations. A statement
about this frontend entry must explicitly condition its scope on entering the
audited fresh ASTSource frontend path; it must not invent a new source guard.

To claim a source-level error contract later, formalize the selected frontend
control fragment and justify its extraction/refinement from the frozen Python,
then compose it with the original wrapper's ordered effects and exception
propagation. Hash equality and observed exceptions alone cannot supply that proof.
