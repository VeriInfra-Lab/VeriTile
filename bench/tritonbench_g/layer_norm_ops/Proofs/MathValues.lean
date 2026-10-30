import bench.tritonbench_g.layer_norm_ops.Proofs.Mathematical
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardIO

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger.Logical (sum)
noncomputable section
set_option maxHeartbeats 3000000

theorem forward_input_math (p : ForwardParameters) (v : ForwardValues ℝ p) (s : State ℝ)
    (bound : ForwardBinding p v s) (covered : p.columns≤p.block) (i : Nat) (hi : i<p.columns) :
    Logical.input p s i=forwardMathInput p v i := by
  let j : TileIndex [p.block] := (⟨i,lt_of_lt_of_le hi covered⟩,())
  have hx : (s.mem p.x (s.pids 0*p.xStride+i)).read .fp32=vector (forwardX v) i := by
    exact (bound.x j hi).trans (vector_index _ j).symm
  unfold Logical.input forwardMathInput
  rw [hx]
  cases hr : p.hasResidual
  · simp only [hr,Bool.false_eq_true,if_false]
  · have hres : (s.mem p.residual (s.pids 0*p.residualStride+i)).read .fp32=
        vector (forwardResidual v) i := by
      exact (bound.residual hr j hi).trans (vector_index _ j).symm
    simp only [hr,if_true,hres]

theorem forward_graph_math (p : ForwardParameters) (v : ForwardValues ℝ p) (s : State ℝ)
    (bound : ForwardBinding p v s) (covered : p.columns≤p.block)
    (o : Fin (forwardOutputs p).length) (j : TileIndex (forwardOutputs p)[o].shape)
    (live : forwardOutputMask p o j) : forwardGraph realAlgebra p v o j=forwardMath p v o j := by
  have hx := forward_input_math p v s bound covered
  have hw (i : Nat) (hi : i<p.columns) : (s.mem p.weight i).read .fp32=vector (forwardWeight v) i := by
    exact (bound.weight (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  have hb (flag : p.hasBias=Bool.true) (i : Nat) (hi : i<p.columns) :
      (s.mem p.bias i).read .fp32=vector (forwardBias v) i := by
    exact (bound.bias flag (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  rcases o with ⟨o,ho⟩
  have n : o<4 := ho
  interval_cases o
  · change forwardY realAlgebra p v j=_
    rw [←forward_y_eq realAlgebra p v s bound j live]
    unfold realAlgebra
    rw [KernelReal.forward_value FastLayernorm.KernelMath.algebra p s covered j live]
    exact MathCongruence.y_eq p.columns p.rms p.hasBias p.epsilon _ _ _ _ _ _ hx hw hb j.1.val live
  · change forwardZ realAlgebra p v j=_
    rw [←forward_z_eq realAlgebra p v s bound]
    unfold realAlgebra
    rw [Logical.z_value]
    simp only [live.2,if_true]
    exact hx j.1.val live.2
  · change forwardMean realAlgebra p v=_
    change p.rms=Bool.false at live
    rw [←forward_mean_eq realAlgebra p v s bound]
    unfold realAlgebra
    rw [Logical.mean_value FastLayernorm.KernelMath.algebra p s covered]
    change sum p.columns (Logical.input p s)/p.columns=Logical.mean p.columns p.rms (forwardMathInput p v)
    rw [MathCongruence.sum_eq _ _ _ hx]
    simp only [Logical.mean,live,Bool.false_eq_true,if_false]
  · change forwardInverse realAlgebra p v=_
    rw [←forward_inverse_eq realAlgebra p v s bound]
    unfold realAlgebra
    rw [Logical.rstd_value FastLayernorm.KernelMath.algebra p s covered]
    exact MathCongruence.inverse_eq p.columns p.rms p.epsilon _ _ hx

theorem backward_graph_math (p : BackwardParameters) (pid : Nat) (v : BackwardValues ℝ p)
    (s : State ℝ) (bound : BackwardBinding p pid v s) (hp : s.pids 0=pid)
    (admissible : KernelReadback.Admissible p) (covered : p.columns≤p.block)
    (o : Fin (backwardOutputs p).length) (j : TileIndex (backwardOutputs p)[o].shape)
    (live : backwardOutputMask p pid o j) :
    backwardGraph realAlgebra p pid v o j=backwardMath p pid v o j := by
  have hx (row : Nat) (hr : rowActive p pid row) (i : Nat) (hi : i<p.columns) :
      KernelReal.readRow s p.x p.xStride row i=vector (backwardX v row) i := by
    exact (bound.x row hr (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  have hw (i : Nat) (hi : i<p.columns) : (s.mem p.weight i).read .fp32=vector (backwardWeight v) i := by
    exact (bound.weight (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  have hdy (row : Nat) (hr : rowActive p pid row) (i : Nat) (hi : i<p.columns) :
      KernelReal.readRow s p.dy p.dyStride row i=vector (backwardDY v row) i := by
    exact (bound.dy row hr (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  have hdres (flag : p.hasDres=Bool.true) (row : Nat) (hr : rowActive p pid row)
      (i : Nat) (hi : i<p.columns) :
      KernelReal.readRow s p.dres p.dresStride row i=vector (backwardDres v row) i := by
    exact (bound.dres flag row hr (⟨i,lt_of_lt_of_le hi covered⟩,()) hi).trans (vector_index _ _).symm
  have hat (row : Nat) (hr : rowActive p pid row) (i : Nat) (hi : i<p.columns) :
      (KernelReal.readRow s p.x p.xStride row i-
        if p.rms then 0 else (s.mem p.mean (row*p.meanStride)).read .fp32)*
        (s.mem p.rstd (row*p.rstdStride)).read .fp32=backwardMathHat p v row i := by
    unfold backwardMathHat
    rw [hx row hr i hi,bound.rstd row hr]
    cases hm : p.rms
    · simp only [hm,Bool.false_eq_true,if_false,bound.mean hm row hr]
    · simp only [hm,if_true]
  have dx (row : Nat) (hr : rowActive p pid row) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
      backwardDX realAlgebra p v row i=backwardMathDX p v row i.1.val := by
    rw [←backward_dx_eq realAlgebra p pid v s bound row hr]
    unfold realAlgebra
    rw [KernelReal.dx_value FastLayernorm.KernelMath.algebra p s covered row i hi]
    exact MathCongruence.dx_eq p.columns p.rms p.hasDres _ _ _ _ _ _ _ _ _ _ _ _
      (hx row hr) hw (hdy row hr) (fun flag => hdres flag row hr)
      (fun flag => bound.mean flag row hr) (bound.rstd row hr) i.1.val hi
  rcases o with ⟨o,ho⟩
  have n : o<5 := ho
  interval_cases o
  · exact dx j.1.val live.1 j.2 live.2
  · change backwardY realAlgebra p v j.1.val j.2=backwardMathY p v j.1.val j.2.1.val
    rw [←backward_y_eq realAlgebra p pid v s bound j.1.val live.2.1 live.1]
    unfold realAlgebra
    rw [KernelReal.y_value FastLayernorm.KernelMath.algebra p s j.1.val j.2 live.2.2]
    unfold KernelReal.rowY backwardMathY
    rw [hat j.1.val live.2.1 j.2.1.val live.2.2,hw j.2.1.val live.2.2]
    cases hb : p.hasBias
    · simp only [hb,Bool.false_eq_true,if_false]
    · have h := bound.bias live.1 hb j.2 live.2.2
      simp only [hb,if_true,h,vector_index]
  · exact dx j.1.val live.2.1 j.2 live.2.2
  · change (backwardPartials realAlgebra p pid v).1 j=backwardMathDW p pid v j.1.val
    rw [←backward_partials_eq realAlgebra p pid v s bound hp]
    unfold realAlgebra
    rw [KernelReal.dw_value FastLayernorm.KernelMath.algebra p s admissible j live]
    unfold KernelReal.partialDW backwardMathDW
    simp only [Backward.start,Backward.stop,hp,rowStart,rowStop]
    apply Finset.sum_congr rfl
    intro row member
    have hr : rowActive p pid row := Finset.mem_Ico.mp member
    change KernelReal.readRow s p.dy p.dyStride row j.1.val*
      ((KernelReal.readRow s p.x p.xStride row j.1.val-
        if p.rms then 0 else (s.mem p.mean (row*p.meanStride)).read .fp32)*
        (s.mem p.rstd (row*p.rstdStride)).read .fp32)=_
    rw [hdy row hr j.1.val live,hat row hr j.1.val live]
  · change (backwardPartials realAlgebra p pid v).2 j=backwardMathDB p pid v j.1.val
    rw [←backward_partials_eq realAlgebra p pid v s bound hp]
    unfold realAlgebra
    rw [KernelReal.db_value FastLayernorm.KernelMath.algebra p s admissible live.1 j live.2]
    unfold KernelReal.partialDB backwardMathDB
    simp only [Backward.start,Backward.stop,hp,rowStart,rowStop]
    apply Finset.sum_congr rfl
    intro row member
    exact hdy row (Finset.mem_Ico.mp member) j.1.val live.2

#axiomsClean forward_graph_math
#axiomsClean backward_graph_math
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
