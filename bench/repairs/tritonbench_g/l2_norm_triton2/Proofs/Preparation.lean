import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Runtime

namespace VeriTile.Bench.Repairs.L2NormTriton2.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World Result Keep Dense Allocated Frame blockSize block_limit)
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

structure PreparedFacts {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M) : Prop where
  keep : Keep ops entry ready
  valid : Valid q.layout
  inputLive : q.snapshot.view.storage∈ready.allocations
  gradientLive : q.gradient.view.storage∈ready.allocations
  outputLive : q.output.view.storage∈ready.allocations
  outputFresh : ∀ old∈entry.allocations,q.output.view.storage.separate old
  gradientRows : q.gradient.rows=q.snapshot.rows
  gradientCols : q.gradient.cols=q.snapshot.cols
  outputRows : q.output.rows=q.snapshot.rows
  outputCols : q.output.cols=q.snapshot.cols
  outputDevice : q.output.view.storage.owner=call.input.storage.owner
  inputAligned : 0<q.snapshot.count → q.snapshot.view.storage.baseByte%16=0
  gradientAligned : 0<q.snapshot.count → q.gradient.view.storage.baseByte%16=0
  outputAligned : q.output.view.storage.baseByte%16=0
  block : q.block=blockSize q.snapshot.cols
  count : q.snapshot.count=call.input.numel
  gradientCount : q.gradient.count=call.gradient.numel
  cols : q.snapshot.cols=call.cols
  values : ∀ gradient i,i<(q.inputFor gradient).count →
    ops.read ready.memory ((q.inputFor gradient).view.storage.key i)=
      logical ops neg entry.memory (call.inputFor gradient) i

private theorem separate_symm (a b : Allocation) (h : a.separate b) : b.separate a := by
  rcases h with h | h | h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem prepare_frame {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call γ) (s : World M)
    (domain : Domain call s) : Keep ops s (prepare rt call s).state := by
  by_cases scalar : call.gradient.shape=[]
  · simpa [prepare,scalar,Result.state] using Keep.refl ops s
  cases hx : rt.normalize call.cols call.input s with
  | error reason after =>
    simpa [prepare,scalar,hx,Result.bind,Result.state] using lib.normalizeFailure _ _ _ _ _ hx
  | ok x s1 =>
    have nx := lib.normalize _ _ _ _ _ hx domain.main.backed domain.main.live (domain.direct Bool.false) domain.positive
    cases hd : rt.normalize call.cols call.gradient s1 with
    | error reason after =>
      simpa [prepare,scalar,hx,hd,Result.bind,Result.state] using
        Keep.trans ops nx.keep (lib.normalizeFailure _ _ _ _ _ hd)
    | ok dy s2 =>
      have nd := lib.normalize _ _ _ _ _ hd domain.gradientBacked (nx.keep.live _ domain.gradientLive) (domain.direct Bool.true) domain.positive
      have both := Keep.trans ops nx.keep nd.keep
      by_cases same : dy.rows=x.rows ∧ dy.cols=x.cols
      · cases hy : rt.forward.empty x.view.storage.owner x.rows x.cols s2 with
        | error reason after =>
          simpa [prepare,scalar,hx,hd,same,hy,Result.bind,Result.state] using
            Keep.trans ops both (lib.shared.emptyFailure _ _ _ _ _ _ hy)
        | ok y s3 =>
          have all := Keep.trans ops both (lib.shared.empty _ _ _ _ _ _ hy).keep
          by_cases limit : x.cols≤blockSize x.cols <;>
            simpa [prepare,scalar,hx,hd,same,hy,limit,Result.bind,Result.state] using all
      · simpa [prepare,scalar,hx,hd,same,Result.bind,Result.state] using both

theorem prepare_sound {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call γ) (s : World M)
    (domain : Domain call s) (q : Prepared) (after : World M)
    (success : prepare rt call s=.ok q after) : PreparedFacts ops neg call s q after := by
  by_cases scalar : call.gradient.shape=[]
  · simp [prepare,scalar] at success
  cases hx : rt.normalize call.cols call.input s with
  | error reason after => simp [prepare,scalar,hx,Result.bind] at success
  | ok x s1 =>
    have nx := lib.normalize _ _ _ _ _ hx domain.main.backed domain.main.live (domain.direct Bool.false) domain.positive
    cases hd : rt.normalize call.cols call.gradient s1 with
    | error reason after => simp [prepare,scalar,hx,hd,Result.bind] at success
    | ok dy s2 =>
      have nd := lib.normalize _ _ _ _ _ hd domain.gradientBacked (nx.keep.live _ domain.gradientLive) (domain.direct Bool.true) domain.positive
      by_cases same : dy.rows=x.rows ∧ dy.cols=x.cols
      · cases hy : rt.forward.empty x.view.storage.owner x.rows x.cols s2 with
        | error reason after => simp [prepare,scalar,hx,hd,same,hy,Result.bind] at success
        | ok y s3 =>
          have ay := lib.shared.empty _ _ _ _ _ _ hy
          by_cases limit : x.cols≤blockSize x.cols
          · simp only [prepare,scalar,if_false,hx,hd,Result.bind,same,if_true,hy,limit,Result.ok.injEq] at success
            obtain ⟨rfl,rfl⟩ := success
            have xlive := nd.keep.live _ nx.live
            have xcapacity : x.rows*x.cols≤y.view.storage.words := by
              rw [←ay.rows,←ay.cols]; exact y.capacity
            have dcapacity : x.rows*x.cols≤dy.view.storage.words := by
              rw [←same.1,←same.2]; exact dy.capacity
            refine ⟨Keep.trans ops (Keep.trans ops nx.keep nd.keep) ay.keep,
              ⟨⟨nx.positive,limit,block_limit _,x.capacity,xcapacity,
                separate_symm _ _ (ay.fresh _ xlive)⟩,
                dcapacity,separate_symm _ _ (ay.fresh _ nd.live)⟩,
              ay.keep.live _ xlive,ay.keep.live _ nd.live,ay.live,
              (fun old ho => ay.fresh old (nd.keep.live old (nx.keep.live old ho))),
              same.1,same.2,ay.rows,ay.cols,ay.device.trans nx.device,
              nx.aligned (domain.direct Bool.false),
              (fun nonempty => nd.aligned (domain.direct Bool.true)
                (by simpa only [Dense.count,same.1,same.2] using nonempty)),ay.aligned,rfl,nx.count,nd.count,nx.cols,?_⟩
            intro gradient i hi
            cases gradient
            · change ops.read after.memory (x.view.storage.key i)=logical ops neg s.memory call.input i
              change i<x.count at hi
              rw [ay.keep.reads _ xlive _ (Allocation.key_contains _ _ (lt_of_lt_of_le hi x.capacity)),
                nd.keep.reads _ nx.live _ (Allocation.key_contains _ _ (lt_of_lt_of_le hi x.capacity))]
              exact nx.values domain.main.backed domain.main.live (domain.direct Bool.false) i hi
            · change ops.read after.memory (dy.view.storage.key i)=logical ops neg s.memory call.gradient i
              change i<dy.count at hi
              rw [ay.keep.reads _ nd.live _ (Allocation.key_contains _ _ (lt_of_lt_of_le hi dy.capacity)),nd.values domain.gradientBacked (nx.keep.live _ domain.gradientLive) (domain.direct Bool.true) i hi]
              unfold logical
              rw [nx.keep.reads _ domain.gradientLive _ (domain.gradientBacked i (by rwa [←nd.count]))]
          · simp [prepare,scalar,hx,hd,same,hy,limit,Result.bind] at success
      · simp [prepare,scalar,hx,hd,same,Result.bind] at success

/-- Logical caller-entry row regrouped by DY's last dimension. -/
def entryRow {M α β γ : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (cols row : Nat) (s : World M) (gradient : Bool) : Nat → α :=
  fun col => if col<cols then logical ops neg s.memory (call.inputFor gradient) (row*cols+col) else default

theorem prepared_inputs {M α β γ : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (s : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call s q ready) (row : Nat) (hr : row<q.snapshot.rows) (gradient : Bool) :
    rowInputs ops q.layout ready.memory row gradient=entryRow ops neg call q.snapshot.cols row s gradient := by
  funext col
  unfold rowInputs entryRow
  change (if col<q.snapshot.cols then _ else _)=(if col<q.snapshot.cols then _ else _)
  by_cases hc : col<q.snapshot.cols
  · simp only [hc,if_true]
    have bound := index_bound q.layout row col hr hc
    cases gradient
    · exact facts.values Bool.false _ bound
    · apply facts.values Bool.true
      simpa only [Prepared.inputFor,if_true,Dense.count,facts.gradientRows,facts.gradientCols] using bound
  · simp [hc]

#axiomsClean prepare_frame
#axiomsClean prepare_sound
#axiomsClean prepared_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Host
