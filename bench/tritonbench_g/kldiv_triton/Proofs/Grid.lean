import bench.tritonbench_g.kldiv_triton.Proofs.HostValues
import bench.tritonbench_g.kldiv_triton.Proofs.HostFrames

namespace VeriTile.Bench.Repairs.KldivTriton.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Physical (ForwardLayout BackwardLayout rowValues jobs job_member)
open Host (point chunk derivative chunk_congr)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 7000000

def rowJobs (cols r : Nat) : List (Nat×Nat) := (List.range cols).map (fun c => (r,c))
def scheduledJobs (cols : Nat) (rows : List Nat) : List (Nat×Nat) := rows.flatMap (rowJobs cols)

def runRows {M α β : Type} (ops : Ops M α β) (key : Nat×Nat → Key)
    (value : M → Nat → Nat → α) (cols : Nat) : List Nat → M → M
  | [],s => s
  | r::rest,s => runRows ops key value cols rest
      (stores ops key (fun rc => value s rc.1 rc.2) (rowJobs cols r) s)

theorem stores_congr {M α β ι : Type} (ops : Ops M α β) (key : ι → Key) (xs ys : ι → α)
    (is : List ι) (s : M) (same : ∀ i∈is,xs i=ys i) : stores ops key xs is s=stores ops key ys is s := by
  induction is generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [stores]
    rw [same i (by simp)]
    exact ih _ (fun j hj => same j (by simp [hj]))

/-- Re-evaluate the full row source against the current physical memory, then
write that row. The stable-input lemma is supplied by the checked source and
physical separation below, not by any runtime output callback. -/
theorem runRows_eq_stores {M α β : Type} (ops : Ops M α β) (key : Nat×Nat → Key)
    (value : M → Nat → Nat → α) (rows cols : Nat) (initial : M)
    (stable : ∀ done,(∀ rc∈done,rc.1<rows ∧ rc.2<cols) → ∀ r c,r<rows → c<cols →
      value (stores ops key (fun rc => value initial rc.1 rc.2) done initial) r c=value initial r c)
    (schedule : List Nat) (valid : ∀ r∈schedule,r<rows)
    (done : List (Nat×Nat)) (doneValid : ∀ rc∈done,rc.1<rows ∧ rc.2<cols) :
    runRows ops key value cols schedule (stores ops key (fun rc => value initial rc.1 rc.2) done initial)=
      stores ops key (fun rc => value initial rc.1 rc.2) (done++scheduledJobs cols schedule) initial := by
  induction schedule generalizing done with
  | nil => simp [runRows,scheduledJobs,stores]
  | cons r rest ih =>
    have hr := valid r (by simp)
    have newValid : ∀ rc∈rowJobs cols r,rc.1<rows ∧ rc.2<cols := by
      intro rc hc
      change rc∈(List.range cols).map (fun c => (r,c)) at hc
      obtain ⟨c,hmem,equal⟩ := List.mem_map.mp hc
      cases equal
      exact ⟨hr,List.mem_range.mp hmem⟩
    have rowEq : stores ops key
        (fun rc => value (stores ops key (fun rc => value initial rc.1 rc.2) done initial) rc.1 rc.2)
        (rowJobs cols r) (stores ops key (fun rc => value initial rc.1 rc.2) done initial)=
        stores ops key (fun rc => value initial rc.1 rc.2) (done++rowJobs cols r) initial := by
      rw [stores_append]
      apply stores_congr
      intro rc hc
      exact stable done doneValid rc.1 rc.2 (newValid rc hc).1 (newValid rc hc).2
    rw [runRows,rowEq]
    rw [ih (fun q hq => valid q (by simp [hq])) (done++rowJobs cols r) (by
      intro rc hc
      rcases List.mem_append.mp hc with hc | hc
      · exact doneValid rc hc
      · exact newValid rc hc)]
    simp only [scheduledJobs,List.flatMap_cons,List.append_assoc]

theorem matrix_store_address (m : Matrix) (unit : m.colStride=1) (r c : Nat) :
    m.key r c=m.view.storage.key (m.view.storageOffset+r*m.rowStride+c) := by
  simp [Matrix.key,Matrix.index,unit]

theorem input_after_stores {M α β : Type} (ops : Ops M α β) (initial : M)
    (input output : Matrix) (backedInput : input.backed) (backedOutput : output.backed)
    (separate : input.view.storage.separate output.view.storage)
    (values : Nat×Nat → α) (done : List (Nat×Nat))
    (valid : ∀ rc∈done,rc.1<output.rows ∧ rc.2<output.cols)
    (r c : Nat) (hr : r < input.rows) (hc : c < input.cols) :
    rowValues ops (stores ops (fun rc => output.key rc.1 rc.2) values done initial) input r c=
      rowValues ops initial input r c := by
  apply read_frame
  intro rc hrc
  exact input.view.storage.spans_separate output.view.storage separate _ _ (backedInput r c hr hc)
    (backedOutput rc.1 rc.2 (valid rc hrc).1 (valid rc hrc).2)

structure ForwardValid (p : ForwardLayout) : Prop where
  predictionUnit : Index.UnitColumns p.prediction
  targetUnit : Index.UnitColumns p.target
  outputUnit : p.output.colStride=1
  targetRows : p.target.rows=p.prediction.rows
  targetCols : p.target.cols=p.prediction.cols
  outputRows : p.output.rows=p.prediction.rows
  outputCols : p.output.cols=if p.reduction=0 then p.prediction.cols else rounds p.prediction.cols p.block
  predictionBacked : p.prediction.backed
  targetBacked : p.target.backed
  outputBacked : p.output.backed
  predictionSeparate : p.prediction.view.storage.separate p.output.view.storage
  targetSeparate : p.target.view.storage.separate p.output.view.storage

theorem forward_stable {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (initial : M) (p : ForwardLayout) (positive : 0<p.block) (valid : ForwardValid p)
    (done : List (Nat×Nat)) (doneValid : ∀ rc∈done,rc.1<p.output.rows ∧ rc.2<p.output.cols)
    (r c : Nat) (hr : r<p.output.rows) (hc : c<p.output.cols) :
    Storage.forwardValue ops adapter
      (stores ops (fun rc => p.output.key rc.1 rc.2)
        (fun rc => Storage.forwardValue ops adapter initial p positive rc.1 rc.2) done initial) p positive r c=
      Storage.forwardValue ops adapter initial p positive r c := by
  have hrY : r<p.prediction.rows := by rw [←valid.outputRows];exact hr
  have hrT : r<p.target.rows := by rw [valid.targetRows];exact hrY
  have ys := input_after_stores ops initial p.prediction p.output valid.predictionBacked valid.outputBacked valid.predictionSeparate
    (fun rc => Storage.forwardValue ops adapter initial p positive rc.1 rc.2) done doneValid r
  have ts := input_after_stores ops initial p.target p.output valid.targetBacked valid.outputBacked valid.targetSeparate
    (fun rc => Storage.forwardValue ops adapter initial p positive rc.1 rc.2) done doneValid r
  by_cases mode : p.reduction=0
  · have col : c<p.prediction.cols := by simpa [valid.outputCols,mode] using hc
    rw [Storage.forward_element ops adapter _ p positive valid.predictionUnit valid.targetUnit mode r c col hrY hrT (by rw [valid.targetCols]; exact col),
      Storage.forward_element ops adapter initial p positive valid.predictionUnit valid.targetUnit mode r c col hrY hrT (by rw [valid.targetCols]; exact col),
      ys c hrY col,ts c hrT (by rw [valid.targetCols];exact col)]
  · have col : c<rounds p.prediction.cols p.block := by simpa [valid.outputCols,mode] using hc
    rw [Storage.forward_chunk ops adapter _ p positive valid.predictionUnit valid.targetUnit mode r c col hrY hrT valid.targetCols,
      Storage.forward_chunk ops adapter initial p positive valid.predictionUnit valid.targetUnit mode r c col hrY hrT valid.targetCols]
    change chunk adapter p.flag p.prediction.cols p.block c _ _=chunk adapter p.flag p.prediction.cols p.block c _ _
    exact chunk_congr adapter p.flag p.prediction.cols p.block c _ _ _ _
      (fun i hi => ys i hrY hi) (fun i hi => ts i hrT (by rw [valid.targetCols];exact hi))

/-- Every row source is evaluated internally at its actual invocation state.
Any valid row order has the same source-derived stores as the snapshot graph. -/
theorem forward_rows {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (initial : M) (p : ForwardLayout) (positive : 0<p.block) (valid : ForwardValid p)
    (schedule : List Nat) (active : ∀ r∈schedule,r<p.output.rows) :
    runRows ops (fun rc => p.output.key rc.1 rc.2) (fun s r c => Storage.forwardValue ops adapter s p positive r c)
      p.output.cols schedule initial=
    stores ops (fun rc => p.output.key rc.1 rc.2) (fun rc => Storage.forwardValue ops adapter initial p positive rc.1 rc.2)
      (scheduledJobs p.output.cols schedule) initial := by
  simpa only [stores,List.nil_append] using runRows_eq_stores ops (fun rc => p.output.key rc.1 rc.2)
    (fun s r c => Storage.forwardValue ops adapter s p positive r c) p.output.rows p.output.cols initial
    (forward_stable ops adapter initial p positive valid) schedule active [] (by simp)

structure BackwardValid (p : BackwardLayout) : Prop where
  targetUnit : Index.UnitColumns p.target
  workUnit : p.work.colStride=1
  targetRows : p.target.rows=p.work.rows
  targetCols : p.target.cols=p.work.cols
  targetBacked : p.target.backed
  workBacked : p.work.backed
  separate : p.target.view.storage.separate p.work.view.storage

theorem backward_rows {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (initial : M) (p : BackwardLayout) (positive : 0<p.block) (valid : BackwardValid p)
    (schedule : List Nat) (active : ∀ r∈schedule,r<p.work.rows) :
    runRows ops (fun rc => p.work.key rc.1 rc.2) (fun s r c => Storage.backwardValue ops adapter s p positive r c)
      p.work.cols schedule initial=
    stores ops (fun rc => p.work.key rc.1 rc.2) (fun rc => Storage.backwardValue ops adapter initial p positive rc.1 rc.2)
      (scheduledJobs p.work.cols schedule) initial := by
  have stable : ∀ (done : List (Nat×Nat)),(∀ rc∈done,rc.1<p.work.rows ∧ rc.2<p.work.cols) → ∀ r c,r<p.work.rows → c<p.work.cols →
      Storage.backwardValue ops adapter (stores ops (fun rc => p.work.key rc.1 rc.2)
        (fun rc => Storage.backwardValue ops adapter initial p positive rc.1 rc.2) done initial) p positive r c=
      Storage.backwardValue ops adapter initial p positive r c := by
    intro done doneValid r c hr hc
    rw [Storage.backward_formula ops adapter _ p positive valid.targetUnit r c hc (by rw [valid.targetRows]; exact hr) (by rw [valid.targetCols]; exact hc),
      Storage.backward_formula ops adapter initial p positive valid.targetUnit r c hc (by rw [valid.targetRows]; exact hr) (by rw [valid.targetCols]; exact hc)]
    rw [input_after_stores ops initial p.target p.work valid.targetBacked valid.workBacked valid.separate _ done doneValid r c
      (by rw [valid.targetRows];exact hr) (by rw [valid.targetCols];exact hc)]
  simpa only [stores,List.nil_append] using runRows_eq_stores ops (fun rc => p.work.key rc.1 rc.2)
    (fun s r c => Storage.backwardValue ops adapter s p positive r c) p.work.rows p.work.cols initial
    stable schedule active [] (by simp)

theorem scheduled_member (cols : Nat) (schedule : List Nat) (r c : Nat) :
    (r,c)∈scheduledJobs cols schedule ↔ r∈schedule ∧ c<cols := by
  simp [scheduledJobs,rowJobs]

/-- Full-grid readback for every permutation of row program IDs, derived from
individual stores and matrix injectivity. No grid-output predicate is assumed. -/
theorem scheduled_readback {M α β : Type} (ops : Ops M α β) (initial : M) (m : Matrix)
    (unique : m.injective) (values : Nat×Nat → α) (schedule : List Nat)
    (permutation : schedule.Perm (List.range m.rows)) (r c : Nat) (hr : r<m.rows) (hc : c<m.cols) :
    ops.read (stores ops (fun rc => m.key rc.1 rc.2) values (scheduledJobs m.cols schedule) initial) (m.key r c)=values (r,c) := by
  apply readback ops _ _ _ initial (r,c)
  · exact (scheduled_member _ _ _ _).mpr ⟨permutation.mem_iff.mpr (List.mem_range.mpr hr),hc⟩
  · intro rc hrc same
    obtain ⟨row,col⟩ := (scheduled_member _ _ _ _).mp hrc
    have rowBound := List.mem_range.mp (permutation.mem_iff.mp row)
    obtain ⟨rEq,cEq⟩ := unique rc.1 rc.2 r c rowBound col hr hc same
    have eq : rc=(r,c) := Prod.ext rEq cEq
    rw [eq]
  · intro rc _ different
    exact m.disjoint _ _ _ _ (Ne.symm different)

theorem scheduled_bytes {M α β : Type} (ops : Ops M α β) (initial : M) (m : Matrix)
    (values : Nat×Nat → α) (schedule : List Nat) (permutation : schedule.Perm (List.range m.rows))
    (owner address : Nat) (outside : ∀ r c,r<m.rows → c<m.cols → ¬(m.key r c).covers owner address) :
    ops.observe (stores ops (fun rc => m.key rc.1 rc.2) values (scheduledJobs m.cols schedule) initial) owner address=
      ops.observe initial owner address := by
  apply bytes_frame
  intro rc hrc
  obtain ⟨row,col⟩ := (scheduled_member _ _ _ _).mp hrc
  exact outside rc.1 rc.2 (List.mem_range.mp (permutation.mem_iff.mp row)) col

/-- Ordinary materialization cannot move an input into an already-live output
allocation. This discharges the row-grid separation condition from local Copy
contracts rather than a new no-alias precondition on caller tensors. -/
theorem prepared_separate {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input prepared : View} {before after : Host.World M}
    (h : Host.Prepared ops neg input prepared before after) (output : Allocation)
    (live : before.live output) (original : input.storage.separate output) : prepared.storage.separate output := by
  rcases h.storage with same | fresh
  · rw [same];exact original
  · exact fresh output live

#axiomsClean scheduled_readback
#axiomsClean scheduled_bytes
#axiomsClean prepared_separate
#axiomsClean matrix_store_address
#axiomsClean forward_rows
#axiomsClean backward_rows
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Grid
