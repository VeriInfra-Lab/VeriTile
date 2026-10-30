import bench.tritonbench_g.matmul_dequantize.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MatmulDequantize.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Complete original typed stage grid, arbitrary permitted schedule, exact
input binding, masks and exterior frame. Logical coordinates retain padding. -/
specification gptq_kernel_correct (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (hBK : p.blockInner=32) (G : Nat) (hG : p.groupSize=32*G)
    (hmode : p.noGroups=Bool.true → p.groupSize=p.inner)
    (hK : p.inner=VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.steps p*32)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p)))) :
    Spec.Real (ReaderIO.gptq p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val)) := by
  apply (ReaderProof.gptq_judgment_iff p valid jobs).mpr
  exact FirstIO.real_implements p valid ha hb hs hz hBK G hG hmode hK jobs schedule

#axiomsClean gptq_kernel_correct

/-- Complete original typed stage grid, arbitrary permitted schedule, exact
input binding, masks and exterior frame. Logical coordinates retain padding. -/
specification s2_kernel_correct (p : FPTypedSecond.Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (hBK : p.blockInner=32) (hsteps : p.steps=(p.inner+31)/32)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    Spec.Real (ReaderIO.s2 p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val)) := by
  apply (ReaderProof.s2_judgment_iff p valid jobs).mpr
  exact SecondIO.real_implements p valid ha hb hs hz hBK hsteps jobs schedule

#axiomsClean s2_kernel_correct

/-- Complete original typed stage grid, arbitrary permitted schedule, exact
input binding, masks and exterior frame. Logical coordinates retain padding. -/
specification dequant32_kernel_correct (p : Dequant32.Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    Spec.Real (ReaderIO.d32 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column)) := by
  apply (ReaderProof.d32_judgment_iff p valid jobs).mpr
  exact D32IO.real_implements p valid hb hs hz jobs schedule

#axiomsClean dequant32_kernel_correct

/-- Complete original typed stage grid, arbitrary permitted schedule, exact
input binding, masks and exterior frame. Logical coordinates retain padding. -/
specification dequant16_kernel_correct (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters)
    (valid : Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry p)))) :
    Spec.Real (ReaderIO.d16 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column)) := by
  apply (ReaderProof.d16_judgment_iff p valid jobs).mpr
  exact D16IO.real_implements p valid hb hs hz jobs schedule

#axiomsClean dequant16_kernel_correct

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantize.Correct
