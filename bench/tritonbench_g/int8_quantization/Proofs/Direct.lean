import bench.tritonbench_g.int8_quantization.Proofs.Contract
import bench.tritonbench_g.int8_quantization.Proofs.SourceMemory
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
abbrev InputTensor := VeriTile.Torch.FP32.Tensor
abbrev TensorLaws := VeriTile.Torch.FP32.TensorLaws
structure Request where
  x : InputTensor
  laws : TensorLaws x
  block : Nat
  query : Bool
  factor : ℝ
  contract : InputContract x.metadata block

def p (c : Request) := parameters c.x.metadata c.block c.query c.factor
def view (c : Request) := VeriTile.Torch.FP32.dataView c.x c.laws c.contract.contiguous
def count (c : Request) := scaleCount c.x.metadata c.block

def zeroScales {α : Type} (zero : α) (n : Nat) (s : State α) : State α :=
  {s with mem:=fun r i=>if r=scaleRegion ∧ i<n then .mk .real zero else s.mem r i}

def typedRun {α : Type} [Inhabited α] (c : Request) (M : Algebra α) (ops : Model.Primitives α) (entry : State α) : State α :=
  if count c=0 then entry
  else if channels c.x.metadata=0 then zeroScales (M.literal (some .fp32) .real 0) (count c) entry
  else Grid.model M ops (p c) (width_positive c.x.metadata c.block) (Schedule.programs 0 (count c)) entry

theorem complete_source {α : Type} [Inhabited α] (c : Request) (M : Algebra α) (ops : Model.Primitives α) (entry : State α)
    (nonempty : 0<count c) (positive : 0<channels c.x.metadata) :
    Grid.launch M ops (p c) entry=some (typedRun c M ops entry) := by
  simp only [typedRun,if_neg (Nat.ne_of_gt nonempty),if_neg (Nat.ne_of_gt positive)]
  exact Grid.whole_source M ops (p c) (width_positive c.x.metadata c.block) entry

theorem typed_output {α : Type} [Inhabited α] (c : Request) (M : Algebra α) (ops : Model.Primitives α) (entry : State α)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c))
    (i : Lane (p c)) (active : Grid.active (p c) w i) :
    (typedRun c M ops entry).mem outRegion (Grid.offset (p c) w i)=
      .mk .int (quantized M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block) (Grid.atProgram entry w) i) := by
  have wb:=Schedule.program_bounds 0 (count c) w member
  have countPos : 0<count c := by omega
  have positive : 0<channels c.x.metadata := by
    have := active.1
    change i.1.val<c.block*channels c.x.metadata at this
    by_contra h
    have zero : channels c.x.metadata=0 := by omega
    simp [zero] at this
  have blocksPos : 0<(p c).blocks := by
    have upper := wb.2.2
    change w.index < 0 + (p c).batches * (p c).blocks at upper
    by_contra h
    have zero : (p c).blocks = 0 := by omega
    simp [zero] at upper
  simp only [typedRun,if_neg (Nat.ne_of_gt countPos),if_neg (Nat.ne_of_gt positive)]
  exact Grid.readback M ops (p c) _ c.contract.blockPositive positive blocksPos _ (Schedule.unique _ _) entry w member i active


theorem program_blocks (c : Request) (w : Schedule.Program)
    (member : w∈Schedule.programs 0 (count c)) : 0 < (p c).blocks := by
  have upper := (Schedule.program_bounds 0 (count c) w member).2.2
  change w.index < 0 + (p c).batches * (p c).blocks at upper
  by_contra h
  have zero : (p c).blocks = 0 := by omega
  simp [zero] at upper

theorem active_channels (c : Request) (w : Schedule.Program) (i : Lane (p c))
    (active : Grid.active (p c) w i) : 0 < channels c.x.metadata := by
  have h := active.1
  change i.1.val < c.block * channels c.x.metadata at h
  by_contra hn
  have hz : channels c.x.metadata = 0 := by omega
  simp [hz] at h

theorem offset_bound (c : Request) (w : Schedule.Program)
    (member : w∈Schedule.programs 0 (count c)) (i : Lane (p c))
    (active : Grid.active (p c) w i) :
    Grid.offset (p c) w i < c.x.metadata.numel := by
  have upper := (Schedule.program_bounds 0 (count c) w member).2.2
  have group : w.index < (p c).batches*(p c).blocks := by simpa using upper
  have batch := (Nat.div_lt_iff_lt_mul (program_blocks c w member)).mpr group
  rw [numel_eq c.x.metadata c.block c.contract]
  exact Geometry.active_offset _ _ _ _ _ _ _ (active_channels c w i active) batch active.2

theorem typed_scale {α : Type} [Inhabited α] (c : Request) (M : Algebra α) (ops : Model.Primitives α)
    (entry : State α) (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) :
    (typedRun c M ops entry).mem scaleRegion w.index = .mk .real
      (if channels c.x.metadata=0 then M.literal (some .fp32) .real 0 else
       savedScale M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block) (Grid.atProgram entry w)) := by
  have upper := (Schedule.program_bounds 0 (count c) w member).2.2
  have countPos : 0 < count c := by omega
  simp only [typedRun,if_neg (Nat.ne_of_gt countPos)]
  by_cases empty : channels c.x.metadata=0
  · simp [empty,zeroScales,show w.index < count c by omega]
  · simp only [if_neg empty]
    exact Grid.scale_readback M ops (p c) _ _ (Schedule.unique _ _) entry w member

structure Heap where
  memory : Words.Memory
  live : Nat → Prop
structure Extension (before after : Heap) : Prop where
  live : ∀ a,before.live a → after.live a
  frame : ∀ a,before.live a → after.memory a=before.memory a
structure Tensor where
  object : Nat
  metadata : TensorMetadata
  base : Nat
  pointer : metadata.dataPtr=BitVec.ofNat 64 base

structure Allocation (c : Request) (before : Heap) where
  output : Tensor
  scales : Tensor
  outputShape : output.metadata.shape=c.x.metadata.shape
  scaleShape : scales.metadata.shape=Direct.scaleShape c.x.metadata c.block
  outputType : output.metadata.dtype=.other "int8"
  scaleType : scales.metadata.dtype=.floating .fp32
  outputDevice : output.metadata.device=c.x.metadata.device
  scaleDevice : scales.metadata.device=c.x.metadata.device
  outputContiguous : output.metadata.isContiguous=Bool.true
  scalesContiguous : scales.metadata.isContiguous=Bool.true
  outputNeg : output.metadata.isNeg=Bool.false
  scalesNeg : scales.metadata.isNeg=Bool.false
  objects : output.object≠scales.object ∧ output.object≠c.x.object ∧ scales.object≠c.x.object
  outputCapacity : output.base+c.x.metadata.numel≤2^64
  scaleCapacity : scales.base+4*count c≤2^64
  scaleAlignment : scales.base%4=0
  separated : ∀ a,Bytes.outputWindow (p c) ⟨(view c).base,output.base,scales.base⟩ a →
    ¬Bytes.scaleWindow (p c) ⟨(view c).base,output.base,scales.base⟩ a
  freshOutput : ∀ a,Bytes.outputWindow (p c) ⟨(view c).base,output.base,scales.base⟩ a → ¬before.live a
  freshScales : ∀ a,Bytes.scaleWindow (p c) ⟨(view c).base,output.base,scales.base⟩ a → ¬before.live a

def layout (c : Request) {before : Heap} (out : Allocation c before) : Bytes.Layout :=
  ⟨(view c).base,out.output.base,out.scales.base⟩

def allocationBounds (c : Request) {before : Heap} (out : Allocation c before) : Bytes.Bounds (p c) (layout c out) := by
  refine ⟨?_,?_,?_,out.separated⟩
  · have span:=(view c).spanFits
    change (view c).base+4*c.x.metadata.numel≤2^64 at span
    simpa [layout,p,parameters,numel_eq _ _ c.contract] using span
  · simpa [layout,p,parameters,numel_eq _ _ c.contract] using out.outputCapacity
  · exact out.scaleCapacity

def after (c : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap) (out : Allocation c before) : Heap where
  memory := Bytes.store (p c) (layout c out) before.memory (typedRun c M ops (Bytes.bind (layout c out) before.memory))
  live := fun a=>before.live a ∨ Bytes.outputWindow (p c) (layout c out) a ∨ Bytes.scaleWindow (p c) (layout c out) a

theorem extension (c : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap) (out : Allocation c before) :
    Extension before (after c M ops before out) := by
  refine ⟨fun a h=>Or.inl h,?_⟩
  intro a live
  apply Bytes.store_frame
  · intro h;exact out.freshOutput a h live
  · intro h;exact out.freshScales a h live

structure Returned where
  output : Tensor
  scales : Tensor
  heap : Heap

def run (c : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap) (out : Allocation c before) : Except InputError Returned := do
  checkTensorInputs c.x.metadata c.block
  return ⟨out.output,out.scales,after c M ops before out⟩

theorem run_success (c : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap) (out : Allocation c before) :
    run c M ops before out=.ok ⟨out.output,out.scales,after c M ops before out⟩ := by
  simp [run,(checkTensorInputs_iff _ _).mpr c.contract]

#axiomsClean complete_source
#axiomsClean extension
end
end VeriTile.Bench.Repairs.Int8Quantization.Direct
