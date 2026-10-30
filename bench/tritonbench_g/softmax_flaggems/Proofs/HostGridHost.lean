import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridTuning

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000

/-- Logical contract of a trusted contiguous conversion / fresh empty_like.
The undefined-load sampler is deliberately unchanged, including on padding. -/
noncomputable def writeSlice (s : BlockState) (R : RegionName) (count : Nat)
    (values : Nat → ℝ) : BlockState :=
  { s with mem := fun r o ↦ if r=R ∧ o<count then .real (values o) else s.mem r o }

theorem writeSlice_frame (s : BlockState) (R : RegionName) (count : Nat) (v : Nat → ℝ) :
    BlockState.WriteWithin (fullSlice R count) s (writeSlice s R count v) := by
  intro r o h
  simp only [writeSlice,fullSlice] at *
  simp [h]

def contiguousStrides : List Nat → List Nat
  | [] => []
  | _::tail => (tail.map fun n ↦ max n 1).prod :: contiguousStrides tail

structure TensorMetadata where
  region : RegionName
  shape : List Nat
  strides : List Nat
  storage : Storage
  cudaDevice : Nat

/-- Ordinary Torch metadata primitives. These map only shape/stride metadata;
they are not kernel executions, values, or success oracles. Singleton strides
need not become canonical after contiguous/empty_like. -/
structure LayoutPrimitives where
  contiguous : List Nat → List Nat → List Nat
  emptyLike : List Nat → List Nat → List Nat

def outputMetadata (R : RegionName) (shape strides : List Nat) (storage : Storage)
    (device : Nat) : TensorMetadata := ⟨R,shape,strides,storage,device⟩

def forwardMetadata (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (R : RegionName) (shape : List Nat) (storage : Storage) (device : Nat) : TensorMetadata :=
  outputMetadata R shape (layouts.emptyLike shape (layouts.contiguous shape inputStrides)) storage device

def gradientMetadata (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (R : RegionName) (shape : List Nat) (storage : Storage) (device : Nat) : TensorMetadata :=
  outputMetadata R shape
    (layouts.emptyLike shape (layouts.emptyLike shape (layouts.contiguous shape inputStrides))) storage device

def outputStorage (input : Storage) (requested : Option Storage) : Storage :=
  requested.getD input

structure SavedContext where
  output : TensorMetadata
  dim : Nat

structure AutogradResult where
  forwardState : BlockState
  finalState : BlockState
  output : TensorMetadata
  saved : SavedContext
  gradient : TensorMetadata
  dimGradient : Option Unit
  dtypeGradient : Option Unit

noncomputable def prepareForward (initial : BlockState) (X P : RegionName) (count : Nat)
    (inputValues emptyOutput : Nat → ℝ) : BlockState :=
  writeSlice (writeSlice initial X count inputValues) P count emptyOutput

/-- A contiguous gradient may alias the saved output. In that case its cells
are already the saved cells; no fictional overwrite with independent values is
performed. For a distinct region, gradValues is the trusted contiguous view. -/
noncomputable def prepareBackward (mid : BlockState) (P G Y : RegionName) (count : Nat)
    (gradValues emptyGradient : Nat → ℝ) : BlockState :=
  writeSlice (if G=P then mid else writeSlice mid G count gradValues) Y count emptyGradient

/-- The actual forward state, with its freshly produced saved output, is passed
to backward. No independent execution snapshots are merged. -/
noncomputable def autograd (shape : List Nat) (ax : Fin shape.length)
    (input : Storage) (requested : Option Storage) (device sms : Nat) (waves : WavesAboveOne)
    (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) : Option AutogradResult := do
  let d := dimensions shape ax
  let outType := outputStorage input requested
  let fcfg := forwardConfig waves d sms
  let count := shape.prod
  let pre := prepareForward initial X P count inputValues emptyOutput
  let mid ← forwardRun input outType P X d fcfg pre
  let beforeBackward := prepareBackward mid P G Y count gradValues emptyGradient
  let final ← tunedBackward outType P G Y d trials choice beforeBackward
  let out := forwardMetadata layouts inputStrides P shape outType device
  return ⟨mid,final,out,⟨out,ax.val⟩,gradientMetadata layouts inputStrides Y shape outType device,none,none⟩

def hostFootprint (X P G Y : RegionName) (count : Nat) : WriteFootprint :=
  fun a ↦ (a.1=X ∨ a.1=P ∨ a.1=G ∨ a.1=Y) ∧ a.2<count

/-- Mathematical source execution composition. The storage parameters select
all existing typed source branches before erasure, including every recorded
autotune trial. This erased helper also covers compiler-rejected profiles; the
source-admitted contract below keeps those rejections explicit. It does not claim
that fp16/bf16/fp32/fp64 rounding equals real arithmetic. -/
specification softmax_autograd_host_grid_real
    (shape : List Nat) (ax : Fin shape.length)
    (input : Storage) (requested : Option Storage) (device sms : Nat) (waves : WavesAboveOne)
    (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ)
    (hM : 0<(shape.take ax.val).prod) (hN : 0<shape[ax.val])
    (hK : 0<(dimensions shape ax).K)
    (hXP : X≠P) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ result,
      autograd shape ax input requested device sms waves layouts inputStrides trials choice X P G Y initial
        inputValues emptyOutput gradValues emptyGradient=some result ∧
      let d := dimensions shape ax
      let pre := prepareForward initial X P shape.prod inputValues emptyOutput
      let beforeBackward := prepareBackward result.forwardState P G Y shape.prod gradValues emptyGradient
      FlatResult P shape.prod (forwardValue pre X d) pre result.forwardState ∧
      FlatResult Y shape.prod (backwardValue beforeBackward P G d (backwardConfig d choice))
        beforeBackward result.finalState ∧
      (∀ o, beforeBackward.readMem P o=result.forwardState.readMem P o) ∧
      (∀ o, result.finalState.readMem P o=result.forwardState.readMem P o) ∧
      result.output=forwardMetadata layouts inputStrides P shape (outputStorage input requested) device ∧
      result.saved=⟨result.output,ax.val⟩ ∧
      result.gradient=gradientMetadata layouts inputStrides Y shape (outputStorage input requested) device ∧
      result.dimGradient=none ∧ result.dtypeGradient=none ∧
      BlockState.WriteWithin (hostFootprint X P G Y shape.prod) initial result.finalState := by
  let d := dimensions shape ax
  let count := shape.prod
  let pre := prepareForward initial X P count inputValues emptyOutput
  have hcount : count=d.M*d.N*d.K := dimensions_numel shape ax hM hN
  have hNd : 0<d.N := hN
  have hKd : 0<d.K := hK
  obtain ⟨mid,hforward,hf⟩ := forward_dispatch input (outputStorage input requested) pre P X d
    (forwardConfig waves d sms) hNd hKd (forwardConfig_positive waves d sms).1
      (forwardConfig_positive waves d sms).2 (forwardConfig_one waves d sms) hXP
  let preb := prepareBackward mid P G Y count gradValues emptyGradient
  obtain ⟨final,hbackward,hb⟩ := tuned_backward_dispatch (outputStorage input requested) preb P G Y d
    trials choice hNd hKd hPY hGY
  have hf' : FlatResult P count (forwardValue pre X d) pre mid := by simpa [hcount] using hf
  have hb' : FlatResult Y count (backwardValue preb P G d (backwardConfig d choice)) preb final := by
    simpa [hcount] using hb
  have saved : ∀ o, preb.readMem P o=mid.readMem P o := by
    intro o
    by_cases hGP : G=P
    · simp [preb,prepareBackward,writeSlice,BlockState.readMem,hGP,hPY]
    · simp [preb,prepareBackward,writeSlice,BlockState.readMem,hGP,Ne.symm hGP,hPY]
  have savedFinal : ∀ o, final.readMem P o=mid.readMem P o := by
    intro o
    have frame := hb'.1 P o (by simp [fullSlice,hPY])
    have hr : final.readMem P o=preb.readMem P o := by
      simp only [BlockState.readMem,← frame]
    exact hr.trans (saved o)
  let out := forwardMetadata layouts inputStrides P shape (outputStorage input requested) device
  let result : AutogradResult := ⟨mid,final,out,⟨out,ax.val⟩,
    gradientMetadata layouts inputStrides Y shape (outputStorage input requested) device,none,none⟩
  refine ⟨result,?_,hf',hb',saved,savedFinal,rfl,rfl,rfl,rfl,rfl,?_⟩
  · have hfwd := hforward
    have hbwd := hbackward
    dsimp only [d,count,pre] at hfwd
    dsimp only [d,count,preb] at hbwd
    simp only [autograd,hfwd,bind,Option.bind,hbwd,Option.pure_def]
    rfl
  · intro R o hn
    have hx : ¬(R=X ∧ o<count) := fun h ↦ hn ⟨Or.inl h.1,h.2⟩
    have hp : ¬(R=P ∧ o<count) := fun h ↦ hn ⟨Or.inr (Or.inl h.1),h.2⟩
    have hg : ¬(R=G ∧ o<count) := fun h ↦ hn ⟨Or.inr (Or.inr (Or.inl h.1)),h.2⟩
    have hy : ¬(R=Y ∧ o<count) := fun h ↦ hn ⟨Or.inr (Or.inr (Or.inr h.1)),h.2⟩
    have preeq : pre.mem R o=initial.mem R o := by simp [pre,prepareForward,writeSlice,hx,hp]
    have prebeq : preb.mem R o=mid.mem R o := by
      by_cases hGP : G=P <;> simp [preb,prepareBackward,writeSlice,hGP,hg,hy]
    exact preeq.symm.trans ((hf'.1 R o hp).trans (prebeq.symm.trans (hb'.1 R o hy)))

inductive HostError where
  | invalidDim | divisionByZero | kernelFailure
  deriving DecidableEq, Repr

/-- Source dim assertion and K computation errors remain observable. K=0 is
not rejected here: the source really selects its inner route, which is unsafe
for a zero-sized post-dimension and is excluded only from the safe theorem. -/
noncomputable def invoke (shape : List Nat) (dim : Int)
    (input : Storage) (requested : Option Storage) (device sms : Nat) (waves : WavesAboveOne)
    (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName) (initial : BlockState)
    (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) : Except HostError AutogradResult :=
  match normalizeDim shape dim with
  | none => .error .invalidDim
  | some ax =>
      let d := dimensions shape ax
      if d.M=0 ∨ d.N=0 then .error .divisionByZero
      else match autograd shape ax input requested device sms waves layouts inputStrides trials choice X P G Y initial
        inputValues emptyOutput gradValues emptyGradient with
        | none => .error .kernelFailure
        | some result => .ok result

theorem zero_tail_dimensions : dimensions [2,0] ⟨0,by decide⟩=⟨1,2,0⟩ := by decide

theorem zero_tail_same_inner_launch (input output : Storage) (Y X : RegionName)
    (c : LaunchConfig) (s : BlockState) :
    forwardRun input output Y X ⟨1,2,0⟩ c s =
      forwardRun input output Y X ⟨1,2,1⟩ c s := by
  simp [forwardRun]

#axiomsClean writeSlice_frame
#axiomsClean softmax_autograd_host_grid_real
#axiomsClean zero_tail_dimensions
#axiomsClean zero_tail_same_inner_launch
#specNonCircular forwardValue avoiding [autograd,invoke,forwardRun,backwardRun,run,exec]
#specNonCircular backwardValue avoiding [autograd,invoke,forwardRun,backwardRun,run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
