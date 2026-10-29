import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem smoothing_zero (o : Options) {p : Parameters} (smoothing : p.smoothing=o.smoothing)
    (flag : p.hasSmoothing=decide (0<o.smoothing)) (nonneg : 0≤o.smoothing) :
    p.hasSmoothing=Bool.false → p.smoothing=0 := by
  intro noSmooth
  have notPositive : ¬0<o.smoothing := by simpa only [flag,decide_eq_false_iff_not] using noSmooth
  rw [smoothing]
  linarith

theorem unsplit_loss_value {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (worldPositive : 0<e.world) (rankBound : e.rank<e.world) (unsplit : ¬(e.world>1 ∨ e.cols>65536))
    (smooth : 0≤o.smoothing) (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col))
    (valid : obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(e.world*e.cols : Nat)))
    (lossPeers lsePeers : Nat → WithBot ℝ) :
    finalLoss k arithmetic lossPeers lsePeers row=
      ↑(Math.loss (e.world*e.cols) x (obs.integer initial labels row) o.ignored o.smoothing o.squareScale) := by
  have world : e.world=1 := by omega
  have rank : e.rank=0 := by omega
  have positive := (forward_positive e).1
  have covered := (forward_width e.cols positive).2.2 (show e.cols≤65536 by omega)
  have source := RealSource.unsplit_loss e.parameters k.launch.ready.memory row x
    (caller_input obs k dtype row hr x caller) (obs.integer initial labels row) positive covered
    (by simp [ForwardData.parameters,forwardParameters,unsplit])
    (smoothing_zero o rfl rfl smooth)
    (by simp [ForwardData.parameters,forwardParameters,rank])
    (by simp [ForwardData.parameters,forwardParameters,world])
    (by simpa only [ForwardData.parameters,forwardParameters,world,Nat.one_mul] using valid)
  have quotient : row/e.rows=0 := Nat.div_eq_of_lt hr
  have remainder : row%e.rows=row := Nat.mod_eq_of_lt hr
  rw [finalLoss,if_neg unsplit]
  simp only [ForwardKernelData.lossValues,quotient,remainder]
  rw [forward_kernel_labels obs k labelDtype row hr]
  simpa only [world,Nat.one_mul] using source

theorem unsplit_lse_value {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (worldPositive : 0<e.world) (unsplit : ¬(e.world>1 ∨ e.cols>65536))
    (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col)) (lsePeers : Nat → WithBot ℝ) :
    finalLSE k arithmetic lsePeers row=↑(Math.lse (e.world*e.cols) x) := by
  have world : e.world=1 := by omega
  have positive := (forward_positive e).1
  have covered := (forward_width e.cols positive).2.2 (show e.cols≤65536 by omega)
  have source := source_lse_slot obs k dtype row 0 hr (forward_positive e).2 x caller
  rw [Math.block_zero_lse e.cols (fwdWidth e.cols) covered] at source
  rw [finalLSE,if_neg unsplit]
  simpa only [world,Nat.one_mul,Nat.zero_mul,Nat.zero_add] using source

/-- Actual unsplit returned and saved tensors satisfy the independent Real
loss and global LSE specification. Shapes and reads come from the host trace. -/
theorem returned_unsplit {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : Library obs rt) (ops : TensorLibrary obs arithmetic rt) (ints : IntegerViews obs)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs arithmetic rt gs gr)
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (worldPositive : 0<e.world) (rankBound : e.rank<e.world) (unsplit : ¬(e.world>1 ∨ e.cols>65536))
    (smooth : 0≤o.smoothing) (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<e.rows → ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x row col))
    (valid : ∀ row, row<e.rows → obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(e.world*e.cols : Nat))) :
    ∀ row, row<e.rows →
      value obs final returned.losses row=
        ↑(Math.loss (e.world*e.cols) (x row) (obs.integer initial labels row) o.ignored o.smoothing o.squareScale) ∧
      value obs final returned.context.lse row=↑(Math.lse (e.world*e.cols) (x row)) := by
  obtain ⟨aggregation⟩ := forward_aggregation e
  have direct : ∀ lossPeers lsePeers, ForwardInputs comms aggregation lossPeers lsePeers := by
    cases aggregation with
    | direct => intro _ _; trivial
    | split flag trace => exact False.elim (unsplit flag)
  let peers : Nat → WithBot ℝ := fun _ => ↑(0 : ℝ)
  have values := forward_values obs arithmetic rt lib ops ints gs gr comms k labelDtype
    (fun many => False.elim (unsplit (Or.inl many))) aggregation peers peers (direct peers peers)
  intro row hr
  have lossBound : row<returned.losses.numel := by simpa [Tensor.numel,values.1.shapeEq] using hr
  have lseBound : row<returned.context.lse.numel := by simpa [Tensor.numel,values.2.1.shapeEq] using hr
  exact ⟨(values.1.values row lossBound).trans
    (unsplit_loss_value obs k dtype labelDtype worldPositive rankBound unsplit smooth row hr (x row) (caller row hr) (valid row hr) peers peers),
    (values.2.1.values row lseBound).trans
    (unsplit_lse_value obs k dtype worldPositive unsplit row hr (x row) (caller row hr) peers)⟩

#axiomsClean unsplit_loss_value
#axiomsClean unsplit_lse_value
#axiomsClean returned_unsplit
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealHost
