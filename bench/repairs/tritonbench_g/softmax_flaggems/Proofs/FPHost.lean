import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPHostCore
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPTuningTrace

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPHost
open VeriTile Triton FP.Structural FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

def Saved (P : RegionName) (r : Result α) := ∀ o, r.finalState.mem P o=r.forwardState.mem P o
def HostFrame (X P G Y : RegionName) (count : Nat) (initial : State α) (r : Result α) :=
  ∀ R o, ¬((R=X ∨ R=P ∨ R=G ∨ R=Y) ∧ o<count) → r.finalState.mem R o=initial.mem R o

theorem reference_saved (A : Algebra α) (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) (hPY : P≠Y) :
    Saved P (reference A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY) := by
  by_cases hz : shape.prod=0
  · simp [Saved,reference,hz,pack]
  · obtain ⟨hM,hN,hK,hcount⟩ := dimensions_positive shape ax hz
    intro o
    simp only [reference,hz,if_false,pack]
    rw [FPTuning.frame A P G Y (dimensions shape ax) hN hK _ _ P o (by simp [hPY])]
    by_cases hGP : G=P
    · simp [prepareBackward,writeSlice,hGP,hPY]
    · simp [prepareBackward,writeSlice,hGP,Ne.symm hGP,hPY]

theorem reference_frame (A : Algebra α) (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) :
    HostFrame X P G Y shape.prod initial (reference A shape ax device sms waves layouts inputStrides
      trials choice X P G Y initial x emptyP g emptyY) := by
  by_cases hz : shape.prod=0
  · intro R o hn; simp [reference,hz,pack]
  · obtain ⟨hM,hN,hK,hcount⟩ := dimensions_positive shape ax hz
    intro R o hn
    have hx : ¬(R=X ∧ o<shape.prod) := fun h => hn ⟨Or.inl h.1,h.2⟩
    have hp : ¬(R=P ∧ o<shape.prod) := fun h => hn ⟨Or.inr (Or.inl h.1),h.2⟩
    have hg : ¬(R=G ∧ o<shape.prod) := fun h => hn ⟨Or.inr (Or.inr (Or.inl h.1)),h.2⟩
    have hy : ¬(R=Y ∧ o<shape.prod) := fun h => hn ⟨Or.inr (Or.inr (Or.inr h.1)),h.2⟩
    simp only [reference,hz,if_false,pack]
    rw [FPTuning.frame A P G Y (dimensions shape ax) hN hK _ _ R o (by simpa [←hcount] using hy)]
    have before : ∀ s : State α, (prepareBackward s P G Y shape.prod g emptyY).mem R o=s.mem R o := by
      intro s
      by_cases hGP : G=P <;> simp [prepareBackward,writeSlice,hGP,hg,hy]
    rw [before]
    rw [FPTrace.forward_frame A X P (dimensions shape ax) (forwardConfig waves (dimensions shape ax) sms)
      hN hK (forwardConfig_positive waves _ sms).1 (forwardConfig_positive waves _ sms).2
      (forwardConfig_one waves _ sms) _ R o (by simpa [←hcount] using hp)]
    simp [prepareForward,writeSlice,hx,hp]

/-- The reference graph itself is execution of the exact active stores. Each
nonempty output cell appears, independently of allocation contents. -/
theorem reference_traces (A : Algebra α) (shape : List Nat) (ax : Fin shape.length)
    (hz : shape.prod≠0) (sms : Nat) (waves : WavesAboveOne)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) :
    let d := dimensions shape ax
    let pre := prepareForward initial X P shape.prod x emptyP
    let f := FPGrid.forwardGraph A X P d (forwardConfig waves d sms) pre
    let before := prepareBackward f P G Y shape.prod g emptyY
    let b := FPTuning.graph A P G Y d (trials++[choice]) before
    ∃ ft bt, f=FPTrace.execute ft pre ∧ b=FPTrace.execute bt before ∧
      (∀ a, a∈FPTrace.events ft ↔ a.1=P ∧ a.2<shape.prod) ∧
      (∀ a, a∈FPTrace.events bt ↔ a.1=Y ∧ a.2<shape.prod) := by
  dsimp only
  obtain ⟨hM,hN,hK,hcount⟩ := dimensions_positive shape ax hz
  refine ⟨FPTrace.forwardTrace A X P _ _ _,FPTuning.trace A P G Y _ _ _,
    FPTrace.forward_graph_trace A X P _ _ _,FPTuning.graph_trace A P G Y _ _ _,?_,?_⟩
  · intro a
    simpa only [hcount] using FPTrace.forward_events_exact A X P (dimensions shape ax)
      (forwardConfig waves (dimensions shape ax) sms) hN hK
      (forwardConfig_positive waves _ sms).1 (forwardConfig_positive waves _ sms).2
      (forwardConfig_one waves _ sms) _ a
  · intro a
    simpa only [hcount] using FPTuning.events_exact A P G Y (dimensions shape ax) hN hK
      (trials++[choice]) (by simp) _ a

theorem whole_typed (A : Algebra α) (shape : List Nat) (dim : Int) (ax : Fin shape.length)
    (hnorm : normalizeDim shape dim=some ax)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) (hPY : P≠Y) :
    ∃ result, invoke A shape dim device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY=.ok result ∧
      Matches result (reference A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
        x emptyP g emptyY) ∧ Saved P result ∧ HostFrame X P G Y shape.prod initial result := by
  obtain ⟨result,he,hm⟩ := whole_model A shape dim ax hnorm device sms waves layouts inputStrides
    trials choice X P G Y initial x emptyP g emptyY
  refine ⟨result,he,hm,?_,?_⟩
  · intro o
    rw [hm.2.1,hm.1]
    exact reference_saved A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY hPY o
  · intro R o hn
    rw [hm.2.1]
    exact reference_frame A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY R o hn

theorem invalid_dimension (A : Algebra α) (shape : List Nat) (dim : Int)
    (hnorm : normalizeDim shape dim=none)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) :
    invoke A shape dim device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY=.error .invalidDim := by simp [invoke,hnorm]

#axiomsClean whole_typed
#axiomsClean reference_traces
#axiomsClean invalid_dimension
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPHost
