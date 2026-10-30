import bench.repairs.tritonbench_g.kldiv_triton.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Contiguous
open Host (blockSize block_positive modeCode)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds outputCount)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

/-- Execute the complete canonical DSL at each evolving physical memory, then
realize its output writes. Source row frames below justify the exact footprint. -/
def executeRows {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (output : Matrix) (region : RegionName)
    (source : Nat → M → Option (State κ)) : List Nat → M → Option M
  | [],memory => some memory
  | r::rest,memory => do
      let final ← source r memory
      let next := stores ops (fun rc => output.key rc.1 rc.2)
        (fun rc => adapter.project ((final.mem region (rc.1*output.rowStride+rc.2)).read .real))
        (Grid.rowJobs output.cols r) memory
      executeRows ops adapter output region source rest next

def forwardRow {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.ForwardLayout) (r : Nat) (memory : M) :=
  FP.Structural.exec adapter.model.alg
    (forwardSource p.prediction.rowStride p.target.rowStride p.output.rowStride
      p.prediction.cols p.block p.flag p.reduction)
    (Storage.forwardState ops adapter memory p r)

def backwardRow {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout) (r : Nat) (memory : M) :=
  FP.Structural.exec adapter.model.alg
    (backwardSource p.work.rowStride p.target.rowStride p.work.cols p.block p.flag)
    (Storage.backwardState ops adapter memory p r)

def executeForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.ForwardLayout) :=
  executeRows ops adapter p.output "loss_ptr" (forwardRow ops adapter p)

def executeBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout) :=
  executeRows ops adapter p.work "input_ptr" (backwardRow ops adapter p)

theorem executeRows_complete {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (output : Matrix) (region : RegionName)
    (source : Nat → M → Option (State κ)) (value : M → Nat → Nat → α)
    (complete : ∀ r memory,∃ final,source r memory=some final ∧ ∀ c,
      adapter.project ((final.mem region (r*output.rowStride+c)).read .real)=value memory r c)
    (schedule : List Nat) (memory : M) :
    executeRows ops adapter output region source schedule memory=
      some (Grid.runRows ops (fun rc => output.key rc.1 rc.2) value output.cols schedule memory) := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons r rest ih =>
    obtain ⟨final,executed,values⟩ := complete r memory
    simp only [executeRows,executed,bind,Option.bind,Grid.runRows]
    have stored : stores ops (fun rc => output.key rc.1 rc.2)
        (fun rc => adapter.project ((final.mem region (rc.1*output.rowStride+rc.2)).read .real))
        (Grid.rowJobs output.cols r) memory=
      stores ops (fun rc => output.key rc.1 rc.2) (fun rc => value memory rc.1 rc.2)
        (Grid.rowJobs output.cols r) memory := by
      apply Grid.stores_congr
      intro rc member
      obtain ⟨c,_,same⟩ := List.mem_map.mp member
      cases same
      exact values c
    rw [stored]
    exact ih _

theorem forward_runs {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.ForwardLayout) (positive : 0<p.block)
    (schedule : List Nat) (memory : M) :
    executeForward ops adapter p schedule memory=some (Grid.runRows ops (fun rc => p.output.key rc.1 rc.2)
      (fun s r c => Storage.forwardValue ops adapter s p positive r c) p.output.cols schedule memory) := by
  apply executeRows_complete
  intro r s
  let result := Tile.sourceRun adapter.model (Storage.forwardState ops adapter s p r)
    p.prediction.rowStride p.target.rowStride p.output.rowStride p.prediction.cols p.block p.flag p.reduction positive
  exact ⟨result.val,result.property,fun _ => rfl⟩

theorem backward_runs {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout) (positive : 0<p.block)
    (schedule : List Nat) (memory : M) :
    executeBackward ops adapter p schedule memory=some (Grid.runRows ops (fun rc => p.work.key rc.1 rc.2)
      (fun s r c => Storage.backwardValue ops adapter s p positive r c) p.work.cols schedule memory) := by
  apply executeRows_complete
  intro r s
  let result := Backward.sourceRun adapter.model (Storage.backwardState ops adapter s p r)
    p.work.rowStride p.target.rowStride p.work.cols p.block p.flag positive
  exact ⟨result.val,result.property,fun _ => rfl⟩

/-- The interpreter itself has no writes outside the row being realized. -/
def RowFrame {κ : Type} (output : Matrix) (region : RegionName) (r : Nat)
    (before after : State κ) : Prop :=
  ∀ name address,name≠region ∨ address<r*output.rowStride ∨ r*output.rowStride+output.cols≤address →
    after.mem name address=before.mem name address

def ForwardSources {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.ForwardLayout) : Prop :=
  ∀ r memory,∃ final,forwardRow ops adapter p r memory=some final ∧
    RowFrame p.output "loss_ptr" r (Storage.forwardState ops adapter memory p r) final

def BackwardSources {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout) : Prop :=
  ∀ r memory,∃ final,backwardRow ops adapter p r memory=some final ∧
    RowFrame p.work "input_ptr" r (Storage.backwardState ops adapter memory p r) final

theorem forward_sources {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.ForwardLayout) (positive : 0<p.block)
    (cols : p.output.cols=outputCount p.prediction.cols p.block p.reduction) : ForwardSources ops adapter p := by
  intro r memory
  let s := Storage.forwardState ops adapter memory p r
  let result := Tile.sourceRun adapter.model s p.prediction.rowStride p.target.rowStride p.output.rowStride
    p.prediction.cols p.block p.flag p.reduction positive
  refine ⟨result.val,result.property,?_⟩
  intro name address outside
  apply Tile.source_frame
  intro hit
  simp only [Tile.RowFootprint,s,Storage.forward_pid,←cols] at hit
  rcases outside with different | lower | upper
  · exact different hit.1
  · omega
  · omega

theorem backward_sources {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (adapter : Storage.Adapter α κ) (p : Physical.BackwardLayout) (positive : 0<p.block) :
    BackwardSources ops adapter p := by
  intro r memory
  let s := Storage.backwardState ops adapter memory p r
  let result := Backward.sourceRun adapter.model s p.work.rowStride p.target.rowStride p.work.cols p.block p.flag positive
  exact ⟨result.val,result.property,fun name address outside =>
    Backward.source_frame adapter.model s p.work.rowStride p.target.rowStride p.work.cols p.block p.flag positive name address outside⟩

/-- Full physical complement, with no list of caller allocations restricting
which raw reads or byte observations are protected. -/
def Frame {M α β : Type} (ops : Ops M α β) (output : Matrix) (before after : M) : Prop :=
  (∀ key,(∀ r c,r<output.rows → c<output.cols → key.disjoint (output.key r c)) →
    ops.read after key=ops.read before key) ∧
  (∀ owner address,(∀ r c,r<output.rows → c<output.cols → ¬(output.key r c).covers owner address) →
    ops.observe after owner address=ops.observe before owner address)

theorem scheduled_frame {M α β : Type} (ops : Ops M α β) (output : Matrix)
    (value : Nat×Nat → α) (schedule : List Nat) (valid : ∀ r∈schedule,r<output.rows) (memory : M) :
    Frame ops output memory (stores ops (fun rc => output.key rc.1 rc.2) value (Grid.scheduledJobs output.cols schedule) memory) := by
  constructor
  · intro key outside
    apply read_frame
    intro rc member
    obtain ⟨hr,hc⟩ := (Grid.scheduled_member _ _ _ _).mp member
    exact outside rc.1 rc.2 (valid _ hr) hc
  · intro owner address outside
    apply bytes_frame
    intro rc member
    obtain ⟨hr,hc⟩ := (Grid.scheduled_member _ _ _ _).mp member
    exact outside rc.1 rc.2 (valid _ hr) hc

def forwardReference {M α β κ : Type} (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (prediction target : Matrix) (flag : Bool) (mode : Math.Mode)
    (memory : M) (r c : Nat) : α :=
  if mode=.none then Host.point adapter flag (row ops neg memory prediction r c) (row ops neg memory target r c)
  else Host.chunk adapter flag prediction.cols (blockSize prediction.cols) c
    (row ops neg memory prediction r) (row ops neg memory target r)

theorem forward_value {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (a b o : Tensor) (prediction target output : Matrix) (flag : Bool) (mode : Math.Mode)
    (domain : ForwardDomain a b o prediction target output mode) (memory : M)
    (r c : Nat) (hr : r<output.rows) (hc : c<output.cols) :
    Storage.forwardValue ops adapter memory (forwardLayout prediction target output flag mode) (block_positive _) r c=
      forwardReference ops neg adapter prediction target flag mode memory r c := by
  obtain ⟨targetRows,targetCols⟩ := domain.predictionBinding.rows_cols domain.targetBinding domain.admitted.shape
  have pr : r<prediction.rows := by rw [←domain.outputRows];exact hr
  have tr : r<target.rows := by rw [targetRows];exact pr
  have py := (domain.predictionBinding.contiguous domain.admitted.left.contiguous).1
  have pt := (domain.targetBinding.contiguous domain.admitted.right.contiguous).1
  have yp := domain.predictionBinding.negative.trans domain.admitted.left.noNeg
  have tp := domain.targetBinding.negative.trans domain.admitted.right.noNeg
  by_cases modeNone : mode=.none
  · have pc : c<prediction.cols := by simpa [domain.outputCols,outputCols,modeNone] using hc
    rw [Storage.forward_element ops adapter memory _ (block_positive _) py pt
      (by simp [forwardLayout,modeCode,modeNone]) r c pc pr tr (by change c<target.cols;rw [targetCols];exact pc)]
    change Host.point adapter flag (Physical.rowValues ops memory prediction r c)
      (Physical.rowValues ops memory target r c)=_
    rw [row_raw ops neg memory prediction yp r c pc,
      row_raw ops neg memory target tp r c (by rw [targetCols];exact pc)]
    simp [forwardReference,modeNone]
  · have cc : c<rounds prediction.cols (blockSize prediction.cols) := by
      simpa [domain.outputCols,outputCols,modeNone] using hc
    have code : modeCode mode≠0 := by cases mode <;> simp_all [modeCode]
    rw [Storage.forward_chunk ops adapter memory _ (block_positive _) py pt code r c cc pr tr targetCols]
    change Host.chunk adapter flag prediction.cols (blockSize prediction.cols) c
      (Physical.rowValues ops memory prediction r) (Physical.rowValues ops memory target r)=_
    simp only [forwardReference,modeNone,if_false]
    apply Host.chunk_congr
    · intro j hj;exact row_raw ops neg memory prediction yp r j hj
    · intro j hj;exact row_raw ops neg memory target tp r j (by rw [targetCols];exact hj)

structure ForwardPost {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (a b o : Tensor) (prediction target output : Matrix)
    (flag : Bool) (mode : Math.Mode) (schedule : List Nat) (expected : Nat → Nat → α) (before after : M) : Prop where
  executes : executeForward ops adapter (forwardLayout prediction target output flag mode) schedule before=some after
  values : ∀ r c,r<output.rows → c<output.cols → row ops neg after output r c=expected r c
  frame : Frame ops output before after
  sources : ForwardSources ops adapter (forwardLayout prediction target output flag mode)
  predictionAddresses : Addresses a prediction
  targetAddresses : Addresses b target
  outputAddresses : Addresses o output
  control : Control prediction

structure BackwardPost {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (a b : Tensor) (destination target : Matrix)
    (flag : Bool) (schedule : List Nat) (expected : Nat → Nat → α) (before after : M) : Prop where
  executes : executeBackward ops adapter (backwardLayout destination target flag) schedule before=some after
  values : ∀ r c,r<destination.rows → c<destination.cols → row ops neg after destination r c=expected r c
  frame : Frame ops destination before after
  sources : BackwardSources ops adapter (backwardLayout destination target flag)
  destinationAddresses : Addresses a destination
  targetAddresses : Addresses b target
  control : Control destination

theorem completeForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (a b o : Tensor) (prediction target output : Matrix) (flag : Bool) (mode : Math.Mode)
    (domain : ForwardDomain a b o prediction target output mode) (dispatch : Dispatch prediction.rows)
    (schedule : List Nat) (permutation : schedule.Perm (List.range output.rows)) (memory : M)
    (expected : Nat → Nat → α) (formula : ∀ r c,r<output.rows → c<output.cols →
      forwardReference ops neg adapter prediction target flag mode memory r c=expected r c) :
    ∃ final,ForwardPost ops neg adapter a b o prediction target output flag mode schedule expected memory final := by
  let p := forwardLayout prediction target output flag mode
  let values := fun rc : Nat×Nat => Storage.forwardValue ops adapter memory p (block_positive _) rc.1 rc.2
  let final := stores ops (fun rc => output.key rc.1 rc.2) values (Grid.scheduledJobs output.cols schedule) memory
  have valid : ∀ r∈schedule,r<output.rows := fun r hr => List.mem_range.mp (permutation.mem_iff.mp hr)
  have execution := forward_runs ops adapter p (block_positive _) schedule memory
  rw [Grid.forward_rows ops adapter memory p (block_positive _) (domain.valid flag) schedule valid] at execution
  obtain ⟨targetRows,_⟩ := domain.predictionBinding.rows_cols domain.targetBinding domain.admitted.shape
  refine ⟨final,execution,?_,scheduled_frame ops output values schedule valid memory,?_,
    addresses domain.predictionBinding domain.admitted.left dispatch,
    addresses domain.targetBinding domain.admitted.right (by rw [targetRows];exact dispatch),
    addresses domain.outputBinding domain.outputAdmitted (by rw [domain.outputRows];exact dispatch),
    control domain.predictionBinding domain.admitted.left dispatch⟩
  · intro r c hr hc
    rw [←row_raw ops neg final output (domain.outputBinding.negative.trans domain.outputAdmitted.noNeg) r c hc]
    change ops.read final (output.key r c)=expected r c
    dsimp only [final]
    rw [Grid.scheduled_readback ops memory output
      (domain.outputBinding.contiguous domain.outputAdmitted.contiguous).2 values schedule permutation r c hr hc]
    exact (forward_value ops neg adapter a b o prediction target output flag mode domain memory r c hr hc).trans
      (formula r c hr hc)
  · apply forward_sources ops adapter p (block_positive _)
    change output.cols=outputCount prediction.cols (blockSize prediction.cols) (modeCode mode)
    rw [domain.outputCols]
    cases mode <;> rfl

theorem completeBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (a b g : Tensor) (destination target : Matrix) (gradient : View) (flag : Bool)
    (domain : BackwardDomain a b g destination target gradient) (dispatch : Dispatch destination.rows)
    (schedule : List Nat) (permutation : schedule.Perm (List.range destination.rows)) (memory : M)
    (expected : Nat → Nat → α) (formula : ∀ r c,r<destination.rows → c<destination.cols →
      Host.derivative adapter flag (row ops neg memory target r c)=expected r c) :
    ∃ final,BackwardPost ops neg adapter a b destination target flag schedule expected memory final := by
  let p := backwardLayout destination target flag
  let values := fun rc : Nat×Nat => Storage.backwardValue ops adapter memory p (block_positive _) rc.1 rc.2
  let final := stores ops (fun rc => destination.key rc.1 rc.2) values (Grid.scheduledJobs destination.cols schedule) memory
  have valid : ∀ r∈schedule,r<destination.rows := fun r hr => List.mem_range.mp (permutation.mem_iff.mp hr)
  have execution := backward_runs ops adapter p (block_positive _) schedule memory
  have rows : Grid.runRows ops (fun rc => p.work.key rc.1 rc.2)
      (fun s r c => Storage.backwardValue ops adapter s p (block_positive _) r c)
      p.work.cols schedule memory=final := by
    exact Contiguous.backward_rows ops adapter a b g ⟨destination,target,gradient,flag⟩ domain.admitted
      domain.destinationBinding domain.targetBinding memory schedule valid
  have executed := execution.trans (congrArg some rows)
  obtain ⟨targetRows,targetCols⟩ := domain.destinationBinding.rows_cols domain.targetBinding domain.admitted.shape
  refine ⟨final,executed,?_,scheduled_frame ops destination values schedule valid memory,
    backward_sources ops adapter p (block_positive _),
    addresses domain.destinationBinding domain.admitted.left dispatch,
    addresses domain.targetBinding domain.admitted.right (by rw [targetRows];exact dispatch),
    control domain.destinationBinding domain.admitted.left dispatch⟩
  intro r c hr hc
  rw [←row_raw ops neg final destination (domain.destinationBinding.negative.trans domain.admitted.left.noNeg) r c hc]
  change ops.read final (destination.key r c)=expected r c
  dsimp only [final]
  rw [Grid.scheduled_readback ops memory destination
    (domain.destinationBinding.contiguous domain.admitted.left.contiguous).2 values schedule permutation r c hr hc]
  change Storage.backwardValue ops adapter memory p (block_positive _) r c=_
  rw [Storage.backward_formula ops adapter memory p (block_positive _)
    (domain.targetBinding.contiguous domain.admitted.right.contiguous).1 r c hc
    (by change r<target.rows;rw [targetRows];exact hr) (by change c<target.cols;rw [targetCols];exact hc)]
  change Host.derivative adapter flag (Physical.rowValues ops memory target r c)=_
  rw [row_raw ops neg memory target (domain.targetBinding.negative.trans domain.admitted.right.noNeg)
    r c (by rw [targetCols];exact hc)]
  exact formula r c hr hc

#axiomsClean forward_runs
#axiomsClean backward_runs
#axiomsClean forward_sources
#axiomsClean backward_sources
#axiomsClean completeForward
#axiomsClean completeBackward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel
