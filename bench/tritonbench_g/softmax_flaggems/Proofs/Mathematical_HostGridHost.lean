import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridTuning

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

def hostFootprint (X P G Y : RegionName) (count : Nat) : WriteFootprint :=
  fun a ↦ (a.1=X ∨ a.1=P ∨ a.1=G ∨ a.1=Y) ∧ a.2<count

inductive HostError where
  | invalidDim | divisionByZero | kernelFailure
  deriving DecidableEq, Repr

theorem zero_tail_dimensions : dimensions [2,0] ⟨0,by decide⟩=⟨1,2,0⟩ := by decide



#axiomsClean writeSlice_frame
#axiomsClean zero_tail_dimensions
#specNonCircular forwardValue avoiding [run,exec]
#specNonCircular backwardValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
