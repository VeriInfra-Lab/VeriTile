import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Runtime

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

structure PreparedFacts {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M) : Prop where
  keep : Keep ops entry ready
  valid : Valid q.layout
  inputLive : q.snapshot.view.storage∈ready.allocations
  outputLive : q.output.view.storage∈ready.allocations
  outputFresh : ∀ old∈entry.allocations,q.output.view.storage.separate old
  outputRows : q.output.rows=q.snapshot.rows
  outputCols : q.output.cols=q.snapshot.cols
  inputDevice : q.snapshot.view.storage.owner=call.input.storage.owner
  outputDevice : q.output.view.storage.owner=call.input.storage.owner
  inputAligned : 0<q.snapshot.count → q.snapshot.view.storage.baseByte%16=0
  outputAligned : q.output.view.storage.baseByte%16=0
  block : q.block=blockSize q.snapshot.cols
  count : q.snapshot.count=call.input.numel
  cols : q.snapshot.cols=call.input.shape.getLastD 0
  values : ∀ i,i<q.snapshot.count →
    ops.read ready.memory (q.snapshot.view.storage.key i)=logical ops neg entry.memory call.input i

theorem prepare_frame {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call γ) (s : World M)
    (domain : Domain call s) : Keep ops s (prepare rt call s).state := by
  cases hn : rt.normalize call.input s with
  | error reason after =>
    simpa [prepare,hn,Result.bind,Result.state] using lib.normalizeFailure _ _ _ _ hn
  | ok x s1 =>
    have normalized := lib.normalize _ _ _ _ hn domain.backed domain.live domain.direct
    cases he : rt.empty x.view.storage.owner x.rows x.cols s1 with
    | error reason after =>
      simpa [prepare,hn,he,Result.bind,Result.state] using
        Keep.trans ops normalized.keep (lib.emptyFailure _ _ _ _ _ _ he)
    | ok y s2 =>
      have keep := Keep.trans ops normalized.keep (lib.empty _ _ _ _ _ _ he).keep
      by_cases limit : x.cols≤blockSize x.cols <;>
        simpa [prepare,hn,he,Result.bind,Result.state,limit] using keep

theorem prepare_sound {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call γ) (s : World M)
    (domain : Domain call s) (q : Prepared) (after : World M)
    (success : prepare rt call s=.ok q after) : PreparedFacts ops neg call s q after := by
  cases hn : rt.normalize call.input s with
  | error reason after => simp [prepare,hn,Result.bind] at success
  | ok x s1 =>
    have normalized := lib.normalize _ _ _ _ hn domain.backed domain.live domain.direct
    cases he : rt.empty x.view.storage.owner x.rows x.cols s1 with
    | error reason after => simp [prepare,hn,he,Result.bind] at success
    | ok y s2 =>
      have allocated := lib.empty _ _ _ _ _ _ he
      by_cases limit : x.cols≤blockSize x.cols
      · simp only [prepare,hn,he,Result.bind,limit,if_pos,Result.ok.injEq] at success
        obtain ⟨rfl,rfl⟩ := success
        have xy : x.view.storage.separate y.view.storage := by
          rcases allocated.fresh x.view.storage normalized.live with h | h | h
          · exact Or.inl (Ne.symm h)
          · exact Or.inr (Or.inr h)
          · exact Or.inr (Or.inl h)
        have inputAligned : 0<x.count → x.view.storage.baseByte%16=0 := by
          intro nonempty
          have key:=congrArg Key.address (normalized.keys 0 nonempty)
          simp only [Allocation.key,Nat.mul_zero,Nat.add_zero] at key
          rw [key]
          exact domain.direct.aligned (by rw [←normalized.count];exact nonempty)
        refine ⟨Keep.trans ops normalized.keep allocated.keep,
          ⟨normalized.positive,limit,block_limit _,x.capacity,?_,xy⟩,
          allocated.keep.live _ normalized.live,allocated.live,
          (fun old hold => allocated.fresh old (normalized.keep.live old hold)),
          allocated.rows,allocated.cols,normalized.device,allocated.device.trans normalized.device,inputAligned,allocated.aligned,rfl,normalized.count,normalized.cols,?_⟩
        · change x.rows*x.cols≤y.view.storage.words
          rw [←allocated.rows,←allocated.cols]
          exact y.capacity
        · intro i hi
          rw [allocated.keep.reads x.view.storage normalized.live _
            (Allocation.key_contains _ _ (lt_of_lt_of_le hi x.capacity))]
          rw [normalized.keys i hi]
          rw [normalized.keep.reads call.input.storage domain.live _ (domain.backed i (by rw [←normalized.count];exact hi))]
          simp [logical,Negation.encode,domain.direct.negative]
      · simp [prepare,hn,he,Result.bind,limit] at success

/-- Independent caller entry row, with only the selected last-dimension
coordinates. Padding is a storage-model default and is never an active load. -/
def entryRow {M α β γ : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (cols row : Nat) (s : World M) : Nat → α :=
  fun col => if col<cols then logical ops neg s.memory call.input (row*cols+col) else default

theorem prepared_inputs {M α β γ : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (s : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call s q ready) (row : Nat) (hr : row<q.snapshot.rows) :
    rowInputs ops q.layout ready.memory row=entryRow ops neg call q.snapshot.cols row s := by
  funext col
  unfold rowInputs entryRow
  change (if col<q.snapshot.cols then _ else _)=(if col<q.snapshot.cols then _ else _)
  by_cases hc : col<q.snapshot.cols
  · simp only [hc,if_true]
    exact facts.values _ (index_bound q.layout row col hr hc)
  · simp [hc]

#axiomsClean prepare_frame
#axiomsClean prepare_sound
#axiomsClean prepared_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
