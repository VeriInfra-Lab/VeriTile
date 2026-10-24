import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Whole

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Logical reads of guarded present upstream gradients; absent do is zero-filled
by the host. These buffers are supplied after forward has returned. -/
def hostInput (p : Ops α) (input : Input α) (T D : Nat) (hasDo : Bool)
    (upstream finalUpstream : Nat → α) : Input α := fun r bh t d =>
  if r="do" then if hasDo then upstream (rowAddr bh T D t d) else p.zero
  else if r="dht" then finalUpstream (stateAddr bh D d) else input r bh t d

def installCotangents (p : Ops α) (s : State α) (hasDo : Bool)
    (upstream finalUpstream : Nat → α) : State α :=
  {s with mem := fun r a => if r="do" then .mk .real (if hasDo then upstream a else p.zero)
    else if r="dht" then .mk .real (finalUpstream a) else s.mem r a}

theorem state_hostInput (p : Ops α) (input : Input α) (T D : Nat) (hasDo U : Bool)
    (upstream finalUpstream : Nat → α) (bh d n : Nat) :
    state p U (hostInput p input T D hasDo upstream finalUpstream) bh d n=state p U input bh d n := by
  induction n with
  | zero => simp [state,seed,hostInput]
  | succ n ih => simp [state,hostInput,ih]

theorem install_inputs (p : Ops α) (s : State α) (input : Input α) (T D : Nat) (hasDo : Bool)
    (upstream finalUpstream : Nat → α) (h : Inputs s input T D) :
    Inputs (installCotangents p s hasDo upstream finalUpstream)
      (hostInput p input T D hasDo upstream finalUpstream) T D := by
  constructor
  · intro r hr bh t d
    rcases hr with rfl|rfl|rfl
    · simpa [read,installCotangents,hostInput] using h.rows "x" (by simp) bh t d
    · simpa [read,installCotangents,hostInput] using h.rows "g" (by simp) bh t d
    · simp [read,installCotangents,hostInput,Cell.read]
  · intro r hr bh d
    rcases hr with rfl|rfl
    · simpa [read,installCotangents,hostInput] using h.slots "h0" (by simp) bh d
    · simp [read,installCotangents,hostInput,Cell.read]

theorem host_result (p : Ops α) (input : Input α) (BH T D BDF BDB : Nat)
    (hF : 0 < BDF) (hB : 0 < BDB) (U S hasDo hasDht : Bool)
    (upstream finalUpstream : Nat → α) (s : State α) (h : Inputs s input T D) :
    ∃ f b,
      gridRun p (forward T D BDF U S).surfaceBody (programs BH D BDF) s=some f ∧
      gridRun p (backward T D BDB U (S && hasDht)).surfaceBody (programs BH D BDB)
        (installCotangents p f hasDo upstream finalUpstream)=some b ∧
      Saved p b input U BH T D ∧
      Gradients p b (hostInput p input T D hasDo upstream finalUpstream) U (S && hasDht) BH T D ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        b.mem "ht" (stateAddr bh D d)=.mk .real (state p U input bh d T)) ∧
      (∀ r a, r≠("do" : RegionName) → r≠("dht" : RegionName) →
        ¬ ForwardRegion BH T D S r a → ¬ BackwardRegion BH T D r a → b.mem r a=s.mem r a) := by
  obtain ⟨f,hf,hI,hO,hH,hframe⟩ := forward_grid p input BH T D BDF hF U S s h
  let c := installCotangents p f hasDo upstream finalUpstream
  let inp := hostInput p input T D hasDo upstream finalUpstream
  have hC : Inputs c inp T D := install_inputs p f input T D hasDo upstream finalUpstream hI
  have hS : Saved p c inp U BH T D := by
    intro bh hb t ht d hd
    simpa [c,inp,installCotangents,state_hostInput] using hO bh hb t ht d hd
  obtain ⟨b,hb,hI,hO,hG,hbf⟩ := backward_grid p inp BH T D BDB hB U (S && hasDht) c hC hS
  refine ⟨f,b,hf,hb,?_,hG,?_,?_⟩
  · intro bh hbh t ht d hd
    simpa [inp,state_hostInput] using hO bh hbh t ht d hd
  · intro hS bh hbh d hd
    rw [hbf "ht" _ (by simp [BackwardRegion])]
    simpa [c,installCotangents] using hH hS bh hbh d hd
  · intro r a hdo hdht hnf hnb
    rw [hbf r a hnb]
    simpa [c,installCotangents,hdo,hdht] using hframe r a hnf

#axiomsClean host_result
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
