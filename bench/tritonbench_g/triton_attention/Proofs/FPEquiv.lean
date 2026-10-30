import bench.tritonbench_g.triton_attention.Proofs.Pair

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
open scoped VeriTile.Spec
noncomputable section

abbrev original := Pair.original
abbrev shifted := Pair.shifted

/-- Actual complete materialized forward/preprocess/backward schedules on the
original initialized-DQ domain. The sole change is exact fixed-128 preprocess
row-base arithmetic; all numerical primitives, casts and replicas are retained. -/
specification grid_equiv (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    original scalar c owners I literal fw pw bw ≡[[]] shifted scalar c owners I literal fw pw bw := by
  exact Spec.FloatingPoint.ofStructural rfl rfl (Pair.equivalent scalar c owners I literal fw pw bw hf hp hb)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquiv
