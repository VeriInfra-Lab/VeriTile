import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardValues
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardReal
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations FloatArray IntegerViews value localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

def arithmetic : Arithmetic (WithBot ℝ) where
  toTensorArithmetic := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.arithmetic
  multiply := Option.map₂ (·*·)

theorem add_coe (a b : ℝ) : arithmetic.add (↑a) (↑b)=↑(a+b) := rfl
theorem scale_coe (a b : ℝ) : arithmetic.scale a (↑b)=↑(a*b) := rfl
theorem multiply_coe (a b : ℝ) : arithmetic.multiply (↑a) (↑b)=↑(a*b) := rfl

variable {σ : Type} {obs : Observations (WithBot ℝ) σ} {rt : Runtime (WithBot ℝ) σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World (WithBot ℝ) σ} {returned : Return}
  {e : ForwardData RealSource.algebra rt o x y rows cols initial returned final}

theorem caller_input (k : ForwardKernelData obs e) (shape : x.shape=[rows,cols])
    (row : Nat) (hr : row<rows) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    RealSource.Input e.parameters.toParameters k.ready.memory row v := by
  intro col hc
  exact (forward_kernel_logits k shape row col hr hc).trans (caller col hc)

theorem source_lse_slot (k : ForwardKernelData obs e) (shape : x.shape=[rows,cols])
    (row block : Nat) (hr : row<rows) (hb : block<e.count) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    k.outputValues .lse (block*rows+row)=
      ↑(VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE cols o.width block (Math.scaled o.logitScale v)) := by
  have coordinates := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.block_coordinates rows row block hr
  simp only [ForwardKernelData.outputValues,coordinates.1,coordinates.2,Grid.roleValue]
  exact RealSource.block_lse e.parameters k.ready.memory row block v (caller_input k shape row hr v caller)
    e.admitted.2.2.2.2.2.1 (VeriTile.Bench.Repairs.CrossEntropy1.Math.block_starts cols o.width block
      e.admitted.2.2.2.2.2.1 (by simpa only [e.countEq] using hb))

theorem source_pair_slot (lib : Library obs arithmetic rt) (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (yLive : initial.live y.region)
    (row block : Nat) (hr : row<rows) (hb : block<e.count) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    (k.outputValues .loss (block*rows+row),k.outputValues .zLoss (block*rows+row))=
      let L := VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE cols o.width block (Math.scaled o.logitScale v)
      let S := VeriTile.Bench.Repairs.CrossEntropy1.Math.blockSum cols o.width block (Math.scaled o.logitScale v)
      let h := Math.smoothing o.smoothing o.hasSmoothing
      let target := obs.integer initial y row
      (↑(if target=o.ignored then 0 else
          (if e.split then 0 else L)-h*S/o.total-
            (if blockOwns e.parameters target block then (1-h)*Math.scaled o.logitScale v (target-o.start).toNat else 0)+
            (if e.split then 0 else o.squareScale*L*L)),
       ↑(if target=o.ignored then 0 else if e.split then 0 else o.squareScale*L*L)) := by
  have coordinates := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.block_coordinates rows row block hr
  simp only [ForwardKernelData.outputValues,coordinates.1,coordinates.2,Grid.roleValue,
    forward_kernel_labels arithmetic lib k yLive row hr,Prod.mk.eta]
  exact RealSource.scalar_outputs e.parameters k.ready.memory row block v (caller_input k shape row hr v caller)
    (obs.integer initial y row) e.admitted.2.2.2.2.2.1
    (VeriTile.Bench.Repairs.CrossEntropy1.Math.block_starts cols o.width block
      e.admitted.2.2.2.2.2.1 (by simpa only [e.countEq] using hb))

theorem reduced_lse (k : ForwardKernelData obs e) (shape : x.shape=[rows,cols])
    (row : Nat) (hr : row<rows) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    localResult arithmetic.logsumexp e.count rows (k.outputValues .lse) row=↑(Math.LSE cols o.logitScale v) := by
  have positive : 0<e.count := by
    rw [e.countEq]
    exact VeriTile.Bench.Repairs.CrossEntropy1.Host.block_count_positive cols o.width e.admitted.1 e.admitted.2.2.2.2.2.1
  have result := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.local_lse e.count rows row positive (k.outputValues .lse)
    (fun block => VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE cols o.width block (Math.scaled o.logitScale v))
    (fun block hb => source_lse_slot k shape row block hr hb v caller)
  rw [e.countEq,VeriTile.Bench.Repairs.CrossEntropy1.Math.local_lse_composition cols o.width e.admitted.2.2.2.2.2.1] at result
  simpa only [e.countEq] using result

theorem reduced_loss (lib : Library obs arithmetic rt) (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (yLive : initial.live y.region) (split : e.split=Bool.true)
    (row : Nat) (hr : row<rows) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    localResult arithmetic.sum e.count rows (k.outputValues .loss) row=
      ↑(if obs.integer initial y row=o.ignored then 0 else
        -Math.smoothing o.smoothing o.hasSmoothing*(∑ j∈Finset.range cols,Math.scaled o.logitScale v j)/o.total-
        Math.targetTerm cols o.start (obs.integer initial y row) (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale v) := by
  have positive : 0<e.count := by
    rw [e.countEq]
    exact VeriTile.Bench.Repairs.CrossEntropy1.Host.block_count_positive cols o.width e.admitted.1 e.admitted.2.2.2.2.2.1
  have slots : ∀ block, block<e.count → k.outputValues .loss (block*rows+row)=
      ↑(Math.partialLoss cols o.total o.width block o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale v) := by
    intro block hb
    have pair := congrArg Prod.fst (source_pair_slot lib k shape yLive row block hr hb v caller)
    simpa only [Math.partialLoss,split,↓reduceIte,zero_sub,add_zero,neg_mul,neg_div,blockOwns,ForwardData.parameters,Host.parameters] using pair
  have result := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.local_sum e.count rows row positive (k.outputValues .loss)
    (fun block => Math.partialLoss cols o.total o.width block o.start (obs.integer initial y row) o.ignored
      (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale v) slots
  rw [e.countEq,Math.partial_partition cols o.total o.width e.admitted.2.2.2.2.2.1] at result
  simpa only [e.countEq] using result

theorem unsplit_count (k : ForwardKernelData obs e) (split : e.split=Bool.false) : e.count=1 ∧ cols≤o.width := by
  have pos : 0<e.count := by
    rw [e.countEq]
    exact VeriTile.Bench.Repairs.CrossEntropy1.Host.block_count_positive cols o.width e.admitted.1 e.admitted.2.2.2.2.2.1
  have flag := e.splitEq
  rw [split] at flag
  have small : ¬e.count>1 := by
    intro many
    simp only [decide_eq_true (show e.count>1 ∨ o.start≠0 ∨ cols≠o.total from Or.inl many),Bool.or_true] at flag
    cases flag
  have one : e.count=1 := by omega
  have covered := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks_cover cols o.width e.admitted.2.2.2.2.2.1
  rw [← e.countEq,one,Nat.one_mul] at covered
  exact ⟨one,covered⟩

/-- Real arithmetic closes the complete host graph against the independent
mathematical target, including ignored and non-owning labels. -/
theorem final_real_values (lib : Library obs arithmetic rt) (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (yLive : initial.live y.region)
    (row : Nat) (hr : row<rows) (v : Nat → ℝ)
    (caller : ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v col)) :
    finalLoss k arithmetic row=↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
      (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale v) ∧
    finalLSE k arithmetic row=↑(Math.LSE cols o.logitScale v) ∧
    finalZ k arithmetic row=↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale v) := by
  cases split : e.split with
  | «true» =>
    have reducedL := reduced_lse k shape row hr v caller
    have reducedX := reduced_loss lib k shape yLive split row hr v caller
    simp only [finalLoss,finalLSE,finalZ,split,↓reduceIte,aggregateLossValue,aggregateZValue,reducedL,reducedX]
    by_cases ignored : obs.integer initial y row=o.ignored
    · simp [ignored,Math.loss,Math.zLoss,arithmetic,VeriTile.Bench.Repairs.CrossEntropy1.RealHost.arithmetic]
    · simp only [ignored,↓reduceIte,Math.loss,Math.zLoss]
      constructor
      · rw [scale_coe,multiply_coe,add_coe,add_coe]
        apply congrArg (fun v : ℝ => (↑v : WithBot ℝ))
        ring
      · exact ⟨trivial,rfl⟩
  | «false» =>
    have count := unsplit_count k split
    have pair := source_pair_slot lib k shape yLive row 0 hr (by omega) v caller
    have lse := source_lse_slot k shape row 0 hr (by omega) v caller
    have owner : blockOwns e.parameters (obs.integer initial y row) 0 ↔
        (o.start : Int)≤obs.integer initial y row ∧ obs.integer initial y row<o.start+(cols : Int) := by
      simp only [blockOwns,ForwardData.parameters,Host.parameters,Nat.zero_mul,Nat.zero_add,Nat.one_mul,Nat.min_eq_left count.2]
      omega
    rw [VeriTile.Bench.Repairs.CrossEntropy1.Math.block_zero_lse cols o.width count.2,
      VeriTile.Bench.Repairs.CrossEntropy1.Math.block_zero_sum cols o.width count.2] at pair
    rw [VeriTile.Bench.Repairs.CrossEntropy1.Math.block_zero_lse cols o.width count.2] at lse
    simp only [Nat.zero_mul,Nat.zero_add,split,Bool.false_eq_true,↓reduceIte,owner] at pair lse
    simp only [finalLoss,finalLSE,finalZ,split,Bool.false_eq_true,↓reduceIte]
    refine ⟨?_,lse,?_⟩
    · have result := congrArg Prod.fst pair
      simp only [Prod.fst] at result
      rw [result]
      unfold Math.loss Math.targetTerm Math.LSE
      by_cases ignored : obs.integer initial y row=o.ignored
      · simp [ignored]
      · simp only [ignored,↓reduceIte]
        apply congrArg (fun v : ℝ => (↑v : WithBot ℝ))
        ring
    · exact congrArg Prod.snd pair

#axiomsClean reduced_lse
#axiomsClean reduced_loss
#axiomsClean final_real_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RealHost
