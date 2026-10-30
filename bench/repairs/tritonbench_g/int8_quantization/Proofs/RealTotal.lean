import bench.repairs.tritonbench_g.int8_quantization.Proofs.RealPair
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceRealTotal
namespace VeriTile.Bench.Repairs.Int8Quantization.RealHost
open VeriTile.Triton FP.Structural Revision002 Words Direct
noncomputable section
set_option maxHeartbeats 600000

/-- Every transformed logical value in this block is in the finite FP32 range.
This predicate inspects original mathematical inputs, not the source's flag
reduction or any output. -/
def validBlock (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) : Prop :=
  Math.FiniteDomain (Grid.parameters (p (ideal c)) w) (Grid.atProgram (entry c storage) w)
local instance (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) :
    Decidable (validBlock c storage w) := Classical.propDecidable _

def totalByte (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) (i : Lane (p c)) : Byte :=
  if validBlock c storage w then targetByte c storage w i else 0

def totalScale (c : Request) (storage : Nat → ℝ) (w : Schedule.Program) : ℝ :=
  if validBlock c storage w then targetScale c storage w else 0

theorem total_output_bytes (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) (i : Lane (p c)) (active : Grid.active (p c) w i) :
    (after c rest store32 storage before out).memory (out.output.base+Grid.offset (p c) w i)=totalByte c storage w i := by
  have bound := Direct.offset_bound c w member i active
  rw [numel_eq _ _ c.contract] at bound
  change Bytes.store (p c) (layout c out) _ _ ((layout c out).outputBase+_) = _
  rw [Bytes.store_output _ _ _ _ _ bound]
  change BitVec.ofInt 8 (((typedRun (ideal c) (Math.algebra rest) Math.primitives (entry c storage)).mem outRegion (Grid.offset (p (ideal c)) w i)).read .int) = _
  rw [typed_output (ideal c) (Math.algebra rest) Math.primitives _ w member i active]
  change BitVec.ofInt 8 (quantized (Math.algebra rest) Math.primitives (Grid.parameters (p (ideal c)) w)
    (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) i) = _
  by_cases finite : validBlock c storage w
  · simp only [totalByte,if_pos finite]
    have real := Math.real_readback rest (Grid.parameters (p (ideal c)) w)
      (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) rfl finite i active
    rw [source_readback (Math.algebra rest) Math.primitives (Grid.parameters (p (ideal c)) w)
      (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) i active] at real
    exact congrArg (BitVec.ofInt 8) (congrArg (Cell.read .int) real)
  · rw [Math.quantized_invalid rest _ _ _ rfl finite]
    simp [totalByte,finite]

theorem total_scale_bytes (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) :
    Bytes.read32 (after c rest store32 storage before out).memory (out.scales.base+4*w.index)=store32 (totalScale c storage w) := by
  have bound : w.index < (p c).batches*(p c).blocks := by
    simpa using (Schedule.program_bounds 0 (count c) w member).2.2
  change Bytes.read32 (Bytes.store (p c) (layout c out) _ _) ((layout c out).scaleBase+_) = _
  rw [Bytes.store_scale _ _ (allocationBounds c out) _ _ _ bound]
  change store32 (((typedRun (ideal c) (Math.algebra rest) Math.primitives (entry c storage)).mem scaleRegion w.index).read .real) = _
  rw [typed_scale (ideal c) (Math.algebra rest) Math.primitives _ w member]
  by_cases zero : channels c.x.metadata=0
  · simp [ideal,zero,Math.algebra,totalScale,targetScale]
  · simp only [show channels (ideal c).x.metadata ≠ 0 from zero,↓reduceIte,Cell.read]
    apply congrArg store32
    by_cases finite : validBlock c storage w
    · simp only [totalScale,if_pos finite,targetScale,if_neg zero]
      have real := Math.real_scale rest (Grid.parameters (p (ideal c)) w)
        (width_positive c.x.metadata c.block) (Grid.atProgram (entry c storage) w) rfl finite
      rw [source_scale] at real
      exact congrArg (Cell.read .real) real
    · rw [Math.scale_invalid rest _ _ _ rfl finite]
      simp [totalScale,finite]

structure TotalValues (c : Request) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (output scales : Tensor) (result : Memory) : Prop where
  outputShape : output.metadata.shape=c.x.metadata.shape
  scaleShape : scales.metadata.shape=Direct.scaleShape c.x.metadata c.block
  outputType : output.metadata.dtype=.other "int8"
  scaleType : scales.metadata.dtype=.floating .fp32
  outputDevice : output.metadata.device=c.x.metadata.device
  scaleDevice : scales.metadata.device=c.x.metadata.device
  bytes : ∀ w, w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c), Grid.active (p c) w i →
    result (output.base+Grid.offset (p c) w i)=totalByte c storage w i
  scales : ∀ w, w∈Schedule.programs 0 (count c) →
    Bytes.read32 result (scales.base+4*w.index)=store32 (totalScale c storage w)

theorem total_values_retained (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) (final : Heap)
    (ext : Extension (after c rest store32 storage before out) final) :
    TotalValues c store32 storage out.output out.scales final.memory := by
  refine ⟨out.outputShape,out.scaleShape,out.outputType,out.scaleType,out.outputDevice,out.scaleDevice,?_,?_⟩
  · intro w hw i hi
    rw [ext.frame _ ?_]
    · exact total_output_bytes c rest store32 storage before out w hw i hi
    · apply Or.inr;apply Or.inl
      have bound := Direct.offset_bound c w hw i hi
      rw [numel_eq _ _ c.contract] at bound
      change out.output.base ≤ out.output.base+Grid.offset (p c) w i ∧
        out.output.base+Grid.offset (p c) w i < out.output.base+(p c).batches*(p c).length*(p c).channels
      exact ⟨Nat.le_add_right _ _,Nat.add_lt_add_left bound _⟩
  · intro w hw
    rw [←total_scale_bytes c rest store32 storage before out w hw]
    funext b
    apply ext.frame
    apply Or.inr;apply Or.inr
    have bound : w.index < count c := by simpa using (Schedule.program_bounds 0 (count c) w hw).2.2
    have hb := b.isLt
    change out.scales.base ≤ out.scales.base+4*w.index+b.val ∧
      out.scales.base+4*w.index+b.val < out.scales.base+4*count c
    omega

theorem pair_total_correct (q k : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q rest store32 storage before qa)) :
    ∃ result, pair q k rest store32 storage before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      TotalValues q store32 storage result.q result.qScale result.heap.memory ∧
      TotalValues k store32 storage result.k result.kScale result.heap.memory ∧
      Extension before result.heap := by
  let mid := after q rest store32 storage before qa
  let final := after k rest store32 storage mid ka
  refine ⟨⟨qa.output,qa.scales,ka.output,ka.scales,final⟩,?_,rfl,rfl,rfl,rfl,?_,?_,?_⟩
  · simp [pair,(checkTensorInputs_iff _ _).mpr q.contract,(checkTensorInputs_iff _ _).mpr k.contract,
      mid,final,Except.bind,bind,pure,Except.pure]
  · exact total_values_retained q rest store32 storage before qa final (extension k rest store32 storage mid ka)
  · exact total_values_retained k rest store32 storage mid ka final ⟨fun _ h=>h,fun _ _=>rfl⟩
  · have qe := extension q rest store32 storage before qa
    have ke := extension k rest store32 storage mid ka
    exact ⟨fun a h=>ke.live a (qe.live a h),fun a h=>(ke.frame a (qe.live a h)).trans (qe.frame a h)⟩

#axiomsClean pair_total_correct
#specNonCircular TotalValues avoiding [pair,after,typedRun,Grid.model,finalState,quantized,savedScale,badValue,flags]
end
end VeriTile.Bench.Repairs.Int8Quantization.RealHost
