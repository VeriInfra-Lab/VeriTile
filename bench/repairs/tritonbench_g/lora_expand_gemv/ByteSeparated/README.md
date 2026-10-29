# lora_expand_gemv

- Name:

kernel_real

Statement:

Spec.Real (ReaderIO.view rest plans g data order ⊨{ReaderIO.labeled g indices} fun _ row column =>
      let values

- [lora_expand_gemv_byte_separated.py](../lora_expand_gemv_byte_separated.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete N*B selected source grid in any permutation, sentinel rows and optional in-place addition included; every Y output is observed through the original address-label Real interpretation.

## Conditions

- Valid g indices remains public: K>0, K<=block<=4096, EVEN_K branch condition, signed sentinel-or-valid IDs, all capacities, output stride/contiguity consequences and original arithmetic/grid bounds.
- coverage is a permutation of List.range g.count.
- LoraIO.domain retains exact address-labeled input cells, including prior Y. LoraIO.realObservation data g is unchanged; output/source evaluation is not assumed.
- Selected ByteSeparated.Contract retains byte-span output/input storage separation, including empty storage spans.

Exact FP program pairs and numerical dependencies are recorded in the [lora_expand_gemv contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
