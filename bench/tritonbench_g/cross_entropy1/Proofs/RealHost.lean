import bench.tritonbench_g.cross_entropy1.Proofs.SavedMetadata
import bench.tritonbench_g.cross_entropy1.Proofs.Aggregation

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def arithmetic : TensorArithmetic (WithBot ℝ) where
  zero := ↑(0 : ℝ)
  add := Option.map₂ (·+·)
  square := Option.map (fun x => x^2)
  scale := fun scale => Option.map (fun x => scale*x)
  sum := List.sum
  logsumexp := fun values => RealSource.algebra.log ((values.map RealSource.algebra.exp).sum)
  rankSum := List.sum

theorem sum_range (count : Nat) (f : Nat → ℝ) :
    ((List.range count).map (fun i => (↑(f i) : WithBot ℝ))).sum=↑(∑ i∈Finset.range count, f i) := by
  induction count with
  | zero => simp
  | succ count ih =>
    simp only [List.range_succ,List.map_append,List.map_singleton,List.sum_append,List.sum_cons,List.sum_nil,
      add_zero,Finset.sum_range_succ,ih,WithBot.coe_add]

theorem logsumexp_range (count : Nat) (f : Nat → ℝ) :
    arithmetic.logsumexp ((List.range count).map (fun i => ↑(f i)))=
      ↑(Real.log (∑ i∈Finset.range count, Real.exp (f i))) := by
  change RealSource.algebra.log ((((List.range count).map (fun i => (↑(f i) : WithBot ℝ))).map RealSource.algebra.exp).sum)=_
  rw [List.map_map]
  change RealSource.algebra.log (((List.range count).map (fun i => (↑(Real.exp (f i)) : WithBot ℝ))).sum)=_
  rw [sum_range]
  rfl

theorem mapped_slots (count rows row : Nat) (v : Nat → WithBot ℝ) (f : Nat → ℝ)
    (slots : ∀ block, block<count → v (block*rows+row)=↑(f block)) :
    (List.range count).map (fun block => v (block*rows+row))=(List.range count).map (fun block => ↑(f block)) := by
  apply List.map_congr_left
  intro block hb
  exact slots block (List.mem_range.mp hb)

theorem local_sum (count rows row : Nat) (positive : 0<count) (v : Nat → WithBot ℝ) (f : Nat → ℝ)
    (slots : ∀ block, block<count → v (block*rows+row)=↑(f block)) :
    localResult arithmetic.sum count rows v row=↑(∑ block∈Finset.range count, f block) := by
  by_cases many : count>1
  · rw [localResult,if_pos many,mapped_slots count rows row v f slots]
    exact sum_range count f
  · have one : count=1 := by omega
    have slot := slots 0 positive
    simpa [localResult,one] using slot

theorem local_lse (count rows row : Nat) (positive : 0<count) (v : Nat → WithBot ℝ) (f : Nat → ℝ)
    (slots : ∀ block, block<count → v (block*rows+row)=↑(f block)) :
    localResult arithmetic.logsumexp count rows v row=↑(Real.log (∑ block∈Finset.range count, Real.exp (f block))) := by
  by_cases many : count>1
  · rw [localResult,if_pos many,mapped_slots count rows row v f slots]
    exact logsumexp_range count f
  · have one : count=1 := by omega
    have slot := slots 0 positive
    simpa [localResult,one,Real.log_exp] using slot

theorem block_coordinates (rows row block : Nat) (hr : row<rows) :
    (block*rows+row)/rows=block ∧ (block*rows+row)%rows=row := by
  have quotient : (block*rows+row)/rows=block := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul,Nat.one_mul]
      omega
  have decomposition := Nat.div_add_mod (block*rows+row) rows
  rw [quotient,Nat.mul_comm rows block] at decomposition
  exact ⟨quotient,by omega⟩

theorem caller_input {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col)) :
    RealSource.Input e.parameters k.launch.ready.memory row x := by
  intro col hc
  exact (forward_kernel_logits obs k dtype row col hr hc).trans (caller col hc)

theorem source_lse_slot {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (row block : Nat) (hr : row<e.rows) (hb : block<e.count) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col)) :
    k.lseValues (block*e.rows+row)=↑(Math.blockLSE e.cols (fwdWidth e.cols) block x) := by
  have coordinates := block_coordinates e.rows row block hr
  simp only [ForwardKernelData.lseValues,coordinates.1,coordinates.2]
  exact RealSource.forward_lse e.parameters k.launch.ready.memory row block x (caller_input obs k dtype row hr x caller)
    (forward_width e.cols (forward_positive e).1).1
    (Math.block_starts e.cols (fwdWidth e.cols) block (forward_width e.cols (forward_positive e).1).1 hb)

theorem source_loss_slot {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (row block : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col))
    (split : e.world>1 ∨ e.cols>65536) (smooth : 0≤o.smoothing) :
    k.lossValues (block*e.rows+row)=
      ↑(RealSource.partialLoss e.parameters x (obs.integer initial labels row) block) := by
  have coordinates := block_coordinates e.rows row block hr
  simp only [ForwardKernelData.lossValues,coordinates.1,coordinates.2,forward_kernel_labels obs k labelDtype row hr]
  apply RealSource.split_loss e.parameters k.launch.ready.memory row block x (caller_input obs k dtype row hr x caller)
  · simp [ForwardData.parameters,forwardParameters,split]
  · intro noSmooth
    have h : ¬0<o.smoothing := by
      simpa only [ForwardData.parameters,forwardParameters,decide_eq_false_iff_not] using noSmooth
    change o.smoothing=0
    linarith

#axiomsClean local_sum
#axiomsClean local_lse
#axiomsClean caller_input
#axiomsClean source_lse_slot
#axiomsClean source_loss_slot
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealHost
