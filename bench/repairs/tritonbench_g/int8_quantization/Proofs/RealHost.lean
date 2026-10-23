import bench.repairs.tritonbench_g.int8_quantization.Proofs.PairResults
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceRealGrid
namespace VeriTile.Bench.Repairs.Int8Quantization.RealHost
open VeriTile.Triton FP.Structural Revision002 Words Direct
noncomputable section
set_option maxHeartbeats 500000

/-- Arbitrary mathematical values at the original input storage addresses.
There is no encoding of these inputs into FP32 words. -/
def input (c : Request) (storage : Nat → ℝ) (i : Nat) : ℝ :=
  let value := storage (c.x.storage.address (c.x.physicalIndex i))
  if c.x.metadata.isNeg then -value else value

def ideal (c : Request) : Request := {c with factor:=RealSpec.factor c.query (channels c.x.metadata)}

def entry (c : Request) (storage : Nat → ℝ) : State ℝ where
  mem := fun r i=> .mk .real (if r=xRegion then storage ((view c).base+4*i) else 0)
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun _=>0
  undef := fun d _ _=>defaultValue d

theorem input_pointer (c : Request) (storage : Nat → ℝ) (i : Nat) (hi : i < c.x.metadata.numel) :
    ((entry c storage).mem xRegion i).read .real=input c storage i := by
  simp only [entry,if_pos rfl,Cell.read,input,c.contract.noNeg,Bool.false_eq_true,↓reduceIte]
  rw [c.laws.contiguous c.contract.contiguous i hi]
  apply congrArg storage
  simp only [view,VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,
    if_neg (by omega : c.x.metadata.numel ≠ 0),VeriTile.Torch.FP32.Storage.address]
  omega

/-- Only the final output storage conversion is supplied: int8 is serialized
as its byte and the FP32 scale is stored using `store32`. This does not turn
real inputs or real intermediate operations into a claimed FP32 execution. -/
def encodeOutputs (store32 : ℝ → Word) (state : State ℝ) : State Word where
  mem := fun r i=> if r=outRegion then .mk .int ((state.mem r i).read .int)
    else .mk .real (store32 ((state.mem r i).read .real))
  regs := fun _ _ _=>none
  pids := state.pids
  numPids := state.numPids
  undef := fun d _ _=>defaultValue d

def after (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) : Heap where
  memory := Bytes.store (p c) (layout c out) before.memory
    (encodeOutputs store32 (typedRun (ideal c) (Math.algebra rest) Math.primitives (entry c storage)))
  live := fun a=>before.live a ∨ Bytes.outputWindow (p c) (layout c out) a ∨ Bytes.scaleWindow (p c) (layout c out) a

theorem extension (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) : Extension before (after c rest store32 storage before out) := by
  refine ⟨fun _ h=>Or.inl h,?_⟩
  intro a live
  apply Bytes.store_frame
  · intro h;exact out.freshOutput a h live
  · intro h;exact out.freshScales a h live

def Finite (c : Request) (storage : Nat → ℝ) : Prop :=
  ∀ w∈Schedule.programs 0 (count c),
    Math.FiniteDomain (Grid.parameters (p (ideal c)) w) (Grid.atProgram (entry c storage) w)

def targetByte (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) (i : Lane (p c)) : Byte :=
  let pp := Grid.parameters (p (ideal c)) w
  let ee := Grid.atProgram (entry c storage) w
  BitVec.ofInt 8 (RealSpec.nearestAway
    (127*(RealSpec.factor c.query (channels c.x.metadata)*Math.raw pp ee i)/Math.blockMaximum pp ee))

def targetScale (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) : ℝ :=
  if channels c.x.metadata=0 then 0 else
    (RealSpec.target c.query (channels c.x.metadata)
      (Math.logicalBlock (Grid.parameters (p (ideal c)) w) (Grid.atProgram (entry c storage) w))).2

theorem output_bytes (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) (finite : Finite c storage)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) (i : Lane (p c)) (active : Grid.active (p c) w i) :
    (after c rest store32 storage before out).memory (out.output.base+Grid.offset (p c) w i)=targetByte c storage w i := by
  have bound := Direct.offset_bound c w member i active
  rw [numel_eq _ _ c.contract] at bound
  change Bytes.store (p c) (layout c out) _ _ ((layout c out).outputBase+_) = _
  rw [Bytes.store_output _ _ _ _ _ bound]
  change BitVec.ofInt 8 (((typedRun (ideal c) (Math.algebra rest) Math.primitives (entry c storage)).mem outRegion (Grid.offset (p (ideal c)) w i)).read .int) = _
  rw [typed_output (ideal c) (Math.algebra rest) Math.primitives _ w member i active]
  change BitVec.ofInt 8 (quantized (Math.algebra rest) Math.primitives (Grid.parameters (p (ideal c)) w)
    (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) i) = _
  have real := Math.real_readback rest (Grid.parameters (p (ideal c)) w)
    (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) rfl (finite w member) i active
  rw [source_readback (Math.algebra rest) Math.primitives (Grid.parameters (p (ideal c)) w)
    (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) i active] at real
  have equal : quantized (Math.algebra rest) Math.primitives (Grid.parameters (p (ideal c)) w)
      (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) i =
      RealSpec.nearestAway (127*(RealSpec.factor c.query (channels c.x.metadata)*
        Math.raw (Grid.parameters (p (ideal c)) w) (Grid.atProgram (entry c storage) w) i)/
        Math.blockMaximum (Grid.parameters (p (ideal c)) w) (Grid.atProgram (entry c storage) w)) := by
    simpa using congrArg (Cell.read .int) real
  exact congrArg (BitVec.ofInt 8) equal

theorem scale_bytes (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) (finite : Finite c storage)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) :
    Bytes.read32 (after c rest store32 storage before out).memory (out.scales.base+4*w.index)=store32 (targetScale c storage w) := by
  have bound : w.index < (p c).batches*(p c).blocks := by
    simpa using (Schedule.program_bounds 0 (count c) w member).2.2
  change Bytes.read32 (Bytes.store (p c) (layout c out) _ _) ((layout c out).scaleBase+_) = _
  rw [Bytes.store_scale _ _ (allocationBounds c out) _ _ _ bound]
  change store32 (((typedRun (ideal c) (Math.algebra rest) Math.primitives (entry c storage)).mem scaleRegion w.index).read .real) = _
  rw [typed_scale (ideal c) (Math.algebra rest) Math.primitives _ w member]
  by_cases zero : channels c.x.metadata=0
  · simp [ideal,zero,Math.algebra,targetScale]
  · simp only [show channels (ideal c).x.metadata ≠ 0 from zero,↓reduceIte,Cell.read,targetScale,if_neg zero]
    apply congrArg store32
    have real := Math.real_scale rest (Grid.parameters (p (ideal c)) w)
      (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) rfl (finite w member)
    rw [source_scale] at real
    simpa using congrArg (Cell.read .real) real

#axiomsClean output_bytes
#axiomsClean scale_bytes
#specNonCircular targetByte avoiding [after,typedRun,Grid.model,finalState,quantized]
#specNonCircular targetScale avoiding [after,typedRun,Grid.model,finalState,savedScale]
end
end VeriTile.Bench.Repairs.Int8Quantization.RealHost
