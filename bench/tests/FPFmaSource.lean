import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.Counterexamples
import VeriTile.Meta.StatementAudit

/-! Explicit `tl.fma` preserves one fused operation, its source precision, and
its operand graphs. The concrete witness separates it from rounded mul/add.
Triton API: https://triton-lang.org/main/python-api/generated/triton.language.fma.html
This is a primitive-model regression, with no native/compiler-conformance claim. -/
namespace VeriTile.Tests.FPFmaSource
open VeriTile.Triton
noncomputable section

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def source (A B C O : RegionName) : ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, C = tl.float32, O = tl.float32
  a = tl.load(A + 0)
  b = tl.load(B + 0)
  c = tl.load(C + 0)
  value = tl.fma(a, b, c)
  tl.store(O + 0, value)
}

/-- The source emits a fused node, not a binary multiplication/addition tree. -/
theorem source_fused_node (A B C O : RegionName) :
    (source A B C O).surfaceBody[3]? = some
      (ComputeStmt.assign .real [] "value"
        (ComputeExpr.fma .fp32 .nil .nil
          (.compute (.alg .fp32 (.ref .real [] "a")))
          (.compute (.alg .fp32 (.ref .real [] "b")))
          (.compute (.alg .fp32 (.ref .real [] "c"))))) := rfl

/-- Missing fused support fails even when there are no output lanes. -/
theorem missing_empty_fma {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α)
    (a : ComputeExpr .real [0]) (b c : ComputeExpr .real [])
    (h : M.fma .fp32 = none) :
    FP.Structural.evalExpr M (.fma .fp32 .scalarR .scalarR a b c) s = none := by
  simp [FP.Structural.evalExpr, h]

/-- Scalar addends are broadcast after evaluating the original operands. -/
theorem scalar_tile_fma {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) (n : Nat)
    (a : ComputeExpr .real [n]) (b c : ComputeExpr .real [])
    (av : FP.Structural.Values α .real [n]) (bv cv : α)
    (f : α → α → α → α)
    (hf : M.fma .fp32 = some f)
    (ha : FP.Structural.evalExpr M a s = some av)
    (hb : FP.Structural.evalExpr M b s = some (fun _ => bv))
    (hc : FP.Structural.evalExpr M c s = some (fun _ => cv)) :
    FP.Structural.evalExpr M (.fma .fp32 .scalarR .scalarR a b c) s =
      some (fun i => f (av i) bv cv) := by
  simp [FP.Structural.evalExpr, hf, ha, hb, hc, Broadcast.leftIndex, Broadcast.rightIndex]

def payload (bits : Nat) : ComputeExpr .real [] :=
  .compute (.const (dtype := .fp32) ⟨BitVec.ofNat 32 bits⟩)

def fusedWitness : ComputeExpr .real [] :=
  .fma .fp32 .nil .nil
    (payload 0x3f800400) (payload 0x3f7ff800) (payload 0xbf800000)

/-- Actual structural FMA evaluation yields -2^-26, while independently
rounded multiplication/addition on the same fp32 words yields zero. -/
theorem fused_not_unfused (rest : FP.Structural.Algebra Word)
    (s : FP.Structural.State Word) :
    (FP.Structural.evalExpr (FP.Software.scalarFP32 rest .ieeeValue) fusedWitness s).map
        (fun values => (values PUnit.unit).toRat?) = some (some (mkRat (-1) 67108864)) ∧
    (FP.add .ieeeValue
      (FP.mul .ieeeValue (FP.Value.ofBits .fp32 0x3f800400)
        (FP.Value.ofBits .fp32 0x3f7ff800))
      (FP.Value.ofBits .fp32 0xbf800000)).toRat? = some 0 := by
  refine ⟨?_, FP.fp32_fma_witness.1⟩
  change some ((FP.fma .ieeeValue (FP.Value.ofBits .fp32 0x3f800400)
    (FP.Value.ofBits .fp32 0x3f7ff800) (FP.Value.ofBits .fp32 0xbf800000)).toRat?) = _
  rw [FP.fp32_fma_witness.2]

/-- error: floating composition requires a structural floating expression -/
#guard_msgs in
example : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = tl.fma(x, x, x) + x
}

#axiomsClean source_fused_node
#axiomsClean missing_empty_fma
#axiomsClean scalar_tile_fma
#axiomsClean fused_not_unfused
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.FPFmaSource
