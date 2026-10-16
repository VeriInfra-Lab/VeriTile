import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedProgram

/-! Whole-state typed equality of the complete source and candidate. The
integer input-pointer product is the sole changed operation. The scan,
reduction, carry arithmetic, casts and storage precision remain identical. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.FPProof
open VeriTile.Triton TypedSource TypedProgram FP.Structural
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000

theorem body_equiv (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat)
    {α : Type} [Inhabited α] (M : Algebra α) (u : State α) :
    run M (parts Bool.false dtype out_dtype s o T BT).body u =
      run M (parts Bool.true dtype out_dtype s o T BT).body u := by
  rw [body_split Bool.false, body_split Bool.true]
  exact TypedContext.run_insert M _ _ _ _ (TypedContext.pointer_step s T BT (offsetOp BT) M) u

/-- No numerical law, successful-execution premise or loop-size restriction
is needed for candidate equality. Separate theorems establish success. -/
theorem whole_state (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat)
    {α : Type} [Inhabited α] (M : Algebra α) (u : State α) :
    FP.Structural.exec M (source Bool.false dtype out_dtype s o T BT) u =
      FP.Structural.exec M (source Bool.true dtype out_dtype s o T BT) u := by
  change run M (source Bool.false dtype out_dtype s o T BT).surfaceBody u =
    run M (source Bool.true dtype out_dtype s o T BT).surfaceBody u
  rw [source_split Bool.false, source_split Bool.true]
  apply TypedContext.run_insert M _ []
  intro v
  exact TypedContext.signed_loop M _ _ _ _ _ _ (body_equiv dtype out_dtype s o T BT M) v

#axiomsClean body_equiv
#axiomsClean whole_state
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.FPProof
