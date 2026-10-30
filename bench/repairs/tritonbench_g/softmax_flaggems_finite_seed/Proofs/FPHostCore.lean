import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTuning
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridHost

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPHost
open VeriTile Triton FP.Structural FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Ordinary tensor copy/allocation ports touch only their valid slice. -/
def writeSlice (s : State α) (R : RegionName) (count : Nat) (v : Nat → α) : State α :=
  {s with mem := fun r o => if r=R ∧ o<count then .mk .real (v o) else s.mem r o}
def prepareForward (s : State α) (X P : RegionName) (count : Nat) (x empty : Nat → α) :=
  writeSlice (writeSlice s X count x) P count empty
def prepareBackward (s : State α) (P G Y : RegionName) (count : Nat) (g empty : Nat → α) :=
  writeSlice (if G=P then s else writeSlice s G count g) Y count empty

theorem writeSlice_congr (s t : State α) (h : s.mem=t.mem)
    (R : RegionName) (count : Nat) (v : Nat → α) :
    (writeSlice s R count v).mem=(writeSlice t R count v).mem := by simp only [writeSlice,h]
theorem prepareBackward_congr (s t : State α) (h : s.mem=t.mem)
    (P G Y : RegionName) (count : Nat) (g empty : Nat → α) :
    (prepareBackward s P G Y count g empty).mem=(prepareBackward t P G Y count g empty).mem := by
  unfold prepareBackward
  split <;> apply writeSlice_congr
  · exact h
  · exact writeSlice_congr s t h G count g

structure Result (α : Type) where
  forwardState : State α
  finalState : State α
  output : TensorMetadata
  saved : SavedContext
  gradient : TensorMetadata
  dimGradient : Option Unit
  dtypeGradient : Option Unit

def pack (shape : List Nat) (ax : Fin shape.length) (device : Nat)
    (layouts : LayoutPrimitives) (inputStrides : List Nat) (P Y : RegionName) (f b : State α) : Result α :=
  let out := forwardMetadata layouts inputStrides P shape .fp32 device
  ⟨f,b,out,⟨out,ax.val⟩,gradientMetadata layouts inputStrides Y shape .fp32 device,none,none⟩

def reference (A : Algebra α) (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) : Result α :=
  if shape.prod=0 then pack shape ax device layouts inputStrides P Y initial initial
  else
    let d := dimensions shape ax
    let pre := prepareForward initial X P shape.prod x emptyP
    let f := FPGrid.forwardGraph A X P d (forwardConfig waves d sms) pre
    let before := prepareBackward f P G Y shape.prod g emptyY
    let b := FPTuning.graph A P G Y d (trials++[choice]) before
    pack shape ax device layouts inputStrides P Y f b

def nonempty (A : Algebra α) (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) : Option (Result α) := do
  let d := dimensions shape ax
  let pre := prepareForward initial X P shape.prod x emptyP
  let f ← FPGrid.forwardRun A X P d (forwardConfig waves d sms) pre
  let before := prepareBackward f P G Y shape.prod g emptyY
  let b ← FPTuning.tuned A P G Y d trials choice before
  return pack shape ax device layouts inputStrides P Y f b

def invoke (A : Algebra α) (shape : List Nat) (dim : Int)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) : Except HostError (Result α) :=
  match normalizeDim shape dim with
  | none => .error .invalidDim
  | some ax =>
    if shape.prod=0 then .ok (pack shape ax device layouts inputStrides P Y initial initial)
    else match nonempty A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY with
      | none => .error .kernelFailure
      | some result => .ok result

def Matches (actual expected : Result α) : Prop :=
  actual.forwardState.mem=expected.forwardState.mem ∧ actual.finalState.mem=expected.finalState.mem ∧
  actual.output=expected.output ∧ actual.saved=expected.saved ∧ actual.gradient=expected.gradient ∧
  actual.dimGradient=none ∧ actual.dtypeGradient=none

theorem whole_model (A : Algebra α) (shape : List Nat) (dim : Int) (ax : Fin shape.length)
    (hnorm : normalizeDim shape dim=some ax)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : State α)
    (x emptyP g emptyY : Nat → α) :
    ∃ result, invoke A shape dim device sms waves layouts inputStrides trials choice X P G Y initial
      x emptyP g emptyY=.ok result ∧
      Matches result (reference A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
        x emptyP g emptyY) := by
  by_cases hz : shape.prod=0
  · refine ⟨pack shape ax device layouts inputStrides P Y initial initial,by simp [invoke,hnorm,hz],?_⟩
    simp [Matches,reference,hz,pack]
  · let d := dimensions shape ax
    let pre := prepareForward initial X P shape.prod x emptyP
    let fref := FPGrid.forwardGraph A X P d (forwardConfig waves d sms) pre
    obtain ⟨f,hf,hm⟩ := FPGrid.forward_whole A X P d (forwardConfig waves d sms)
      (forwardConfig_positive waves d sms).1 pre pre rfl
    let before := prepareBackward f P G Y shape.prod g emptyY
    let bref := prepareBackward fref P G Y shape.prod g emptyY
    obtain ⟨b,hb,hbm⟩ := FPTuning.tuned_complete A P G Y d trials choice before bref
      (prepareBackward_congr f fref hm P G Y shape.prod g emptyY)
    refine ⟨pack shape ax device layouts inputStrides P Y f b,?_,?_⟩
    · simp only [invoke,hnorm,hz,if_false]
      have he : nonempty A shape ax device sms waves layouts inputStrides trials choice X P G Y initial
          x emptyP g emptyY=some (pack shape ax device layouts inputStrides P Y f b) := by
        change ((FPGrid.forwardRun A X P d (forwardConfig waves d sms) pre).bind _) = _
        rw [hf]
        change ((FPTuning.tuned A P G Y d trials choice before).bind _) = _
        rw [hb]
        rfl
      rw [he]
    · simp only [Matches,reference,hz,if_false,pack]
      refine ⟨hm,hbm,?_,?_,?_,?_,?_⟩ <;> trivial

theorem dimensions_positive (shape : List Nat) (ax : Fin shape.length) (hz : shape.prod≠0) :
    0<(dimensions shape ax).M ∧ 0<(dimensions shape ax).N ∧ 0<(dimensions shape ax).K ∧
      shape.prod=(dimensions shape ax).M*(dimensions shape ax).N*(dimensions shape ax).K := by
  have hp := Nat.pos_of_ne_zero hz
  rw [shape_factorization shape ax] at hp
  have hm := Nat.pos_of_mul_pos_right (Nat.pos_of_mul_pos_right hp)
  have hn := Nat.pos_of_mul_pos_left (Nat.pos_of_mul_pos_right hp)
  refine ⟨hm,hn,?_,dimensions_numel shape ax hm hn⟩
  rw [dimensions_suffix shape ax hm hn]
  exact Nat.pos_of_mul_pos_left hp

#axiomsClean whole_model
#axiomsClean dimensions_positive
#specNonCircular reference avoiding [invoke,nonempty,FPGrid.run,FP.Structural.exec,FPTuning.run,FPTuning.tuned]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPHost
