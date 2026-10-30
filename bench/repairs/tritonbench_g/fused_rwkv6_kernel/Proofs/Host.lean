import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Execution
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000
variable {α : Type}

theorem partial_address_bound (c : Shape) (z i t f : Nat)
    (hz : z<c.Z) (hi : i<c.NK) (ht : t<c.T) (hf : f<c.V) :
    partialAddress c z i t f<c.NK*c.Z*c.T*c.V := by
  have a : i*c.Z+z<c.NK*c.Z := by nlinarith
  have b : (i*c.Z+z)*c.T+t<c.NK*c.Z*c.T := by nlinarith
  dsimp [partialAddress]
  nlinarith

theorem partial_readback (a : Ops α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (scale : α) (m : Memory α) (z i t f : Nat)
    (hz : z<c.Z) (hi : i<c.NK) (ht : t<c.T) (hf : f<c.V) :
    runGrid a c plan scale (programs c) m .o (partialAddress c z i t f)=
      a.storeCast (scalarPartial a c plan (inputs c m) scale z i (visit c t) f) := by
  rw [complete_grid a c hc plan scale m]
  have hout : Output c .o (partialAddress c z i t f) :=
    Or.inl ⟨rfl,partial_address_bound c z i t f hz hi ht hf⟩
  rw [if_pos hout]
  have dec := partial_decode c z i t f hz ht hf
  simp only [target,dec.1,dec.2.1,dec.2.2.1,dec.2.2.2]

theorem tree_congr_on {a : Ops α} {t : Tree} {f g : Nat → α}
    (h : ∀ i∈t.leaves, f i=g i) : t.eval a.zero a.add f=t.eval a.zero a.add g := by
  induction t with
  | zero => rfl
  | leaf i => exact h i (by simp [Tree.leaves])
  | add x y hx hy =>
    simp only [Tree.eval]
    rw [hx (fun i hi => h i (by simp [Tree.leaves,hi])),hy (fun i hi => h i (by simp [Tree.leaves,hi]))]

theorem legal_leaf {t : Tree} {n i : Nat} (h : t.legal n) (hi : i∈t.leaves) : i<n :=
  List.mem_range.1 (h.mem_iff.1 hi)

theorem result_bound (c : Shape) (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    resultAddress c z t f<c.Z*c.T*c.V := by
  have a : z*c.T+t<c.Z*c.T := by nlinarith
  dsimp [resultAddress]
  nlinarith

/-- Host sum consumes the memory actually produced by every launched key block. -/
theorem returned_value (a : Ops α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (scale : α) (m : Memory α) (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    forward a c plan scale m .result (resultAddress c z t f)=
      scalarOutput a c plan (inputs c m) scale z t f := by
  have hb := result_bound c z t f hz ht hf
  have dec := result_decode c z t f ht hf
  simp only [forward,hostSum,reduceCtorEq,true_and,hb,if_true,dec.1,dec.2.1,dec.2.2,
    hostValue,scalarOutput]
  congr 1
  apply tree_congr_on (a:=a)
  intro i hi
  exact partial_readback a c hc plan scale m z i t f hz (legal_leaf (plan.hostLegal z t f) hi) ht hf

theorem terminal_value (a : Ops α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (scale : α) (m : Memory α) (hs : c.final=true) (z e f : Nat)
    (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) :
    forward a c plan scale m .ht (stateAddress c z e f)=
      a.storeCast (scalarState a c (inputs c m) z e f c.T) := by
  have hb : stateAddress c z e f<c.Z*c.K*c.V := by
    have : z*c.K+e<c.Z*c.K := by nlinarith
    dsimp [stateAddress]
    nlinarith
  simp only [forward,hostSum,reduceCtorEq,false_and,if_false]
  rw [complete_grid a c hc plan scale m]
  have hout : Output c .ht (stateAddress c z e f) := Or.inr ⟨hs,rfl,hb⟩
  rw [if_pos hout]
  have dec := state_decode c z e f he hf
  simp only [target,dec.1,dec.2.1,dec.2.2]

def WholeOutput (c : Shape) (r : Region) (b : Nat) : Prop :=
  Output c r b ∨ (r=.result ∧ b<c.Z*c.T*c.V)

theorem complete_frame (a : Ops α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (scale : α) (m : Memory α) (r : Region) (b : Nat) (h : ¬WholeOutput c r b) :
    forward a c plan scale m r b=m r b := by
  have ho : ¬Output c r b := fun ho => h (Or.inl ho)
  have hr : ¬(r=.result ∧ b<c.Z*c.T*c.V) := fun hr => h (Or.inr hr)
  simp only [forward,hostSum,if_neg hr,complete_grid a c hc plan scale m,if_neg ho]

/-- Full wrapper result including the optional tensor and the saved forward context.
The original Python scale remains distinct from the FP32 launch scalar.
The source has no backward method; the saved context does not introduce one. -/
structure Return (α : Type) where
  output : Nat → Nat → Nat → α
  finalState : Option (Nat → Nat → Nat → α)
  saved : Input α
  savedScale : ℝ
  savedReverse : Bool

def wrapper {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (pythonScale : ℝ) (scale : α) (m : Memory α) : Return α :=
  let f := forward a c plan scale m
  ⟨fun z t v => f .result (resultAddress c z t v),
    if c.final then some (fun z e v => f .ht (stateAddress c z e v)) else none,
    inputs c m,pythonScale,c.reverse⟩

def helperScale (requested : ℝ) (pythonDefault : ℝ) := if requested = -1 then pythonDefault else requested

def helper {α : Type} (a : Ops α) (c : Shape) (plan : Plans {c with reverse:=false})
    (requested : ℝ) (pythonDefault : ℝ)
    (bindFP32 : ℝ → α) (m : Memory α) : Return α :=
  let forwardShape := {c with reverse:=false}
  wrapper a forwardShape plan
    (helperScale requested pythonDefault) (bindFP32 (helperScale requested pythonDefault)) m

/-- Helper's sentinel and reverse default, separate from native scalar binding. -/
theorem helper_default (a : Ops α) (c : Shape) (plan : Plans {c with reverse:=false}) (defaultScale : ℝ)
    (bind : ℝ → α) (m : Memory α) :
    (helper a c plan (-1) defaultScale bind m).savedScale=defaultScale ∧
    (helper a c plan (-1) defaultScale bind m).savedReverse=false := by
  simp [helper,helperScale,wrapper]

theorem wrapper_context (a : Ops α) (c : Shape) (plan : Plans c) (pythonScale : ℝ) (scale : α) (m : Memory α) :
    (wrapper a c plan pythonScale scale m).saved=inputs c m ∧
    (wrapper a c plan pythonScale scale m).savedScale=pythonScale ∧
    (wrapper a c plan pythonScale scale m).savedReverse=c.reverse ∧
    ((wrapper a c plan pythonScale scale m).finalState.isSome=c.final) := by
  cases hs : c.final <;> simp [wrapper,hs]

end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
