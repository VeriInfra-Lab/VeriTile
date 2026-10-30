import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.FP
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Numerical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNative

abbrev value (w : BitVec 32) : FP.Value .fp32 := ⟨w⟩
def Finite (w : BitVec 32) : Prop := (value w).toRat?.isSome
instance (w : BitVec 32) : Decidable (Finite w) := inferInstanceAs (Decidable (_ = Bool.true))

/-- Historical primitive-model adapter, retained for optional host results.
This is an execution-semantics boundary, not an experimental rewrite rule.
RNE, input/result FTZ disabled; results may overflow. The current kernel
FPEquiv entry is unconditional in these native primitive functions. -/
structure LocalFP32 (native : Scalars) : Prop where
  fma : ∀ a b c, Finite a → Finite b → Finite c →
    native.fma a b c = (FP.fma FP.Config.ieeeValue (value a) (value b) (value c)).bits
  add : ∀ a b, Finite a → Finite b →
    native.add a b = (FP.add FP.Config.ieeeValue (value a) (value b)).bits

/-- Each finite-operand instruction uses concrete IEEE binary32 software
semantics. Other instructions retain the same ordered native special-value
behavior, so this statement never presumes canonical NaN payloads. -/
def finiteReference (native : Scalars) : Scalars where
  fma := fun a b c => if Finite a ∧ Finite b ∧ Finite c then
    (FP.fma FP.Config.ieeeValue (value a) (value b) (value c)).bits else native.fma a b c
  add := fun a b => if Finite a ∧ Finite b then
    (FP.add FP.Config.ieeeValue (value a) (value b)).bits else native.add a b

theorem native_eq_reference (native : Scalars) (localLaws : LocalFP32 native) :
    native = finiteReference native := by
  have hf : native.fma = (finiteReference native).fma := by
    funext a b c
    dsimp only [finiteReference]
    split
    · rename_i h
      exact localLaws.fma a b c h.1 h.2.1 h.2.2
    · rfl
  have ha : native.add = (finiteReference native).add := by
    funext a b
    dsimp only [finiteReference]
    split
    · rename_i h
      exact localLaws.add a b h.1 h.2
    · rfl
  exact congrArg₂ Scalars.mk hf ha

#axiomsClean native_eq_reference
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixVectorMultip.Numerical
