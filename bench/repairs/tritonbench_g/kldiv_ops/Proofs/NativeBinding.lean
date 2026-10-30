import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Launch

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
noncomputable section

 def Dense.native (t : Tensor) (d : Dense t) : Native.Dense :=
  ⟨denseView t d,d.nonnull⟩

theorem dense_count_bound (t : Tensor) (d : Dense t) : t.numel<2^62 := Native.count_bound (d.native t)

theorem dense_pointer_exact (t : Tensor) (d : Dense t) (i : Nat) (hi : i<t.numel) (k : Fin 4) :
    (BitVec.ofNat 64 (t.address i+k.val)).toNat=t.address i+k.val := by
  simpa only [Dense.native,dense_address t d i hi] using Native.pointer_exact (d.native t) i hi k

theorem flat_index (rows cols row col : Nat) (hr : row<rows) (hc : col<cols) : row*cols+col<rows*cols := by
  have hm := Nat.mul_le_mul_right cols (show row+1≤rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at hm
  omega

/-- Capacity is the actual active-load footprint, not a shape-equality gate or
compiler failure assumption. Larger/rank-different dense targets are admitted. -/
theorem ForwardPlan.active_pointers {rows cols : Nat} {prediction target output : Tensor} {red : Reduction}
    (plan : ForwardPlan rows cols prediction target output red)
    (targetCapacity : rows*cols≤target.numel) (row col : Nat) (hr : row<rows) (hc : col<cols) (k : Fin 4) :
    (BitVec.ofNat 64 (prediction.address (row*cols+col)+k.val)).toNat=prediction.address (row*cols+col)+k.val ∧
    (BitVec.ofNat 64 (target.address (row*cols+col)+k.val)).toNat=target.address (row*cols+col)+k.val ∧
    (BitVec.ofNat 64 (output.address (if red=.none then row*cols+col else row)+k.val)).toNat=
      output.address (if red=.none then row*cols+col else row)+k.val := by
  have hi := flat_index rows cols row col hr hc
  refine ⟨dense_pointer_exact prediction plan.predictionDense _ (by simpa [Tensor.numel,plan.predictionShape] using hi) k,
    dense_pointer_exact target plan.targetDense _ (by omega) k,?_⟩
  apply dense_pointer_exact output plan.outputDense
  cases red <;> simpa [Tensor.numel,plan.outputShape,Reduction.outputShape] using (by assumption)

theorem ForwardPlan.reduced_pointer {rows cols : Nat} {prediction target output : Tensor} {red : Reduction}
    (plan : ForwardPlan rows cols prediction target output red) (reduced : red≠.none)
    (row : Nat) (hr : row<rows) (k : Fin 4) :
    (BitVec.ofNat 64 (output.address row+k.val)).toNat=output.address row+k.val := by
  apply dense_pointer_exact output plan.outputDense
  simpa [Tensor.numel,plan.outputShape,Reduction.outputShape,reduced] using hr

/-- Every i64 loop/lane intermediate follows from the actual allocated input
and the unchanged block cap. No caller-supplied native-size cap occurs here. -/
theorem ForwardPlan.loop_integers {rows cols : Nat} {prediction target output : Tensor} {red : Reduction}
    (plan : ForwardPlan rows cols prediction target output red)
    (row start lane : Nat) (hr : row<rows) (hs : start<cols) (hl : lane<blockSize cols) :
    cols<2^62 ∧ (BitVec.ofNat 64 (start+lane)).toInt=(start+lane : Nat) ∧
      (BitVec.ofNat 64 (start+blockSize cols)).toInt=(start+blockSize cols : Nat) := by
  have hm : (plan.predictionDense.native prediction).view.words=rows*cols := by
    simp [Dense.native,denseView,Tensor.numel,plan.predictionShape]
  have bounds := Native.loop_intermediates (plan.predictionDense.native prediction) rows cols row start
    (blockSize cols) lane hm hr hs (block_capped cols) hl
  exact ⟨bounds.1,Native.signed_exact _ bounds.2.1,Native.signed_exact _ bounds.2.2⟩

theorem BackwardPlan.active_pointers {rows cols : Nat} {target output : Tensor}
    (plan : BackwardPlan rows cols target output) (outputCapacity : rows*cols≤output.numel)
    (row col : Nat) (hr : row<rows) (hc : col<cols) (k : Fin 4) :
    (BitVec.ofNat 64 (target.address (row*cols+col)+k.val)).toNat=target.address (row*cols+col)+k.val ∧
    (BitVec.ofNat 64 (output.address (row*cols+col)+k.val)).toNat=output.address (row*cols+col)+k.val := by
  have hi := flat_index rows cols row col hr hc
  exact ⟨dense_pointer_exact target plan.targetDense _ (by simpa [Tensor.numel,plan.targetShape] using hi) k,
    dense_pointer_exact output plan.outputDense _ (by omega) k⟩

theorem BackwardPlan.loop_integers {rows cols : Nat} {target output : Tensor}
    (plan : BackwardPlan rows cols target output)
    (row start lane : Nat) (hr : row<rows) (hs : start<cols) (hl : lane<blockSize cols) :
    cols<2^62 ∧ (BitVec.ofNat 64 (start+lane)).toInt=(start+lane : Nat) ∧
      (BitVec.ofNat 64 (start+blockSize cols)).toInt=(start+blockSize cols : Nat) := by
  have hm : (plan.targetDense.native target).view.words=rows*cols := by
    simp [Dense.native,denseView,Tensor.numel,plan.targetShape]
  have bounds := Native.loop_intermediates (plan.targetDense.native target) rows cols row start
    (blockSize cols) lane hm hr hs (block_capped cols) hl
  exact ⟨bounds.1,Native.signed_exact _ bounds.2.1,Native.signed_exact _ bounds.2.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
