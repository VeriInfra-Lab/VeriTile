import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.WholePair

/-! Carry-first normalizer optimization of the actual runtime-scalar source.
Both FP16 and signed-byte E5M2 paths retain their masks, conversions, seeded
MMA, reduction trees and full ordinary grids. The fixed source-selected native
model contracts both normalizer spellings to the same ordered FMA. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquiv
open VeriTile Triton FP.Structural Source Spec
noncomputable section

/-- Original `sum(p,1) + l_prev` program and its full physical CTA grid. -/
abbrev original (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) := WholePair.original A C arg B c

/-- Existing carry-first optimization `l_prev + sum(p,1)`, with the same raw
scalar specialization and all original input/output storage precisions. -/
abbrev candidate (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) := WholePair.candidate A C arg B c

/-- Both actual complete programs execute safely, all observed FP16 output
cells agree, and both exact exterior frames hold. The floating inputs are
arbitrary words; only reached E5M2 input roles require signed-byte encoding.
The empty experimental rule set uses the fixed native selection model and
does not assert agreement between arbitrary hardware and that model. -/
specification attention_kernel_equiv (B : Nat) (A : Target.Ops α)
    (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument) (c : Config)
    (_supported : c.D=16 ∨ c.D=32 ∨ c.D=64 ∨ c.D=128) (hD : 0<c.D) (hN : 0<c.N)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    original A C arg B c ≡[[]] candidate A C arg B c :=
  WholePair.equivalent A C arg B c hD hN hinj

#axiomsClean attention_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquiv
