import bench.repairs.tritonbench_g.int8_quantization.Proofs.RealHost
namespace VeriTile.Bench.Repairs.Int8Quantization.RealHost
open VeriTile.Triton FP.Structural Revision002 Words Direct
noncomputable section
set_option maxHeartbeats 400000

structure Values (c : Request) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (output scales : Tensor) (result : Memory) : Prop where
  outputShape : output.metadata.shape=c.x.metadata.shape
  scaleShape : scales.metadata.shape=Direct.scaleShape c.x.metadata c.block
  outputType : output.metadata.dtype=.other "int8"
  scaleType : scales.metadata.dtype=.floating .fp32
  outputDevice : output.metadata.device=c.x.metadata.device
  scaleDevice : scales.metadata.device=c.x.metadata.device
  bytes : ∀ w, w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c), Grid.active (p c) w i →
    result (output.base+Grid.offset (p c) w i)=targetByte c storage w i
  scales : ∀ w, w∈Schedule.programs 0 (count c) →
    Bytes.read32 result (scales.base+4*w.index)=store32 (targetScale c storage w)

theorem values_retained (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) (finite : Finite c storage) (final : Heap)
    (ext : Extension (after c rest store32 storage before out) final) :
    Values c store32 storage out.output out.scales final.memory := by
  refine ⟨out.outputShape,out.scaleShape,out.outputType,out.scaleType,out.outputDevice,out.scaleDevice,?_,?_⟩
  · intro w hw i hi
    rw [ext.frame _ ?_]
    · exact output_bytes c rest store32 storage before out finite w hw i hi
    · apply Or.inr;apply Or.inl
      have bound := Direct.offset_bound c w hw i hi
      rw [numel_eq _ _ c.contract] at bound
      change out.output.base ≤ out.output.base+Grid.offset (p c) w i ∧
        out.output.base+Grid.offset (p c) w i < out.output.base+(p c).batches*(p c).length*(p c).channels
      exact ⟨Nat.le_add_right _ _,Nat.add_lt_add_left bound _⟩
  · intro w hw
    rw [←scale_bytes c rest store32 storage before out finite w hw]
    funext b
    apply ext.frame
    apply Or.inr;apply Or.inr
    have bound : w.index < count c := by simpa using (Schedule.program_bounds 0 (count c) w hw).2.2
    have hb := b.isLt
    change out.scales.base ≤ out.scales.base+4*w.index+b.val ∧
      out.scales.base+4*w.index+b.val < out.scales.base+4*count c
    omega

def pair (q k : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q rest store32 storage before qa)) : Except InputError PairReturned := do
  checkTensorInputs q.x.metadata q.block
  let middle := after q rest store32 storage before qa
  checkTensorInputs k.x.metadata k.block
  return ⟨qa.output,qa.scales,ka.output,ka.scales,after k rest store32 storage middle ka⟩

theorem pair_correct (q k : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q rest store32 storage before qa))
    (finiteQ : Finite q storage) (finiteK : Finite k storage) :
    ∃ result, pair q k rest store32 storage before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      Values q store32 storage result.q result.qScale result.heap.memory ∧
      Values k store32 storage result.k result.kScale result.heap.memory ∧
      Extension before result.heap := by
  let mid := after q rest store32 storage before qa
  let final := after k rest store32 storage mid ka
  refine ⟨⟨qa.output,qa.scales,ka.output,ka.scales,final⟩,?_,rfl,rfl,rfl,rfl,?_,?_,?_⟩
  · simp [pair,(checkTensorInputs_iff _ _).mpr q.contract,(checkTensorInputs_iff _ _).mpr k.contract,
      mid,final,Except.bind,bind,pure,Except.pure]
  · exact values_retained q rest store32 storage before qa finiteQ final (extension k rest store32 storage mid ka)
  · exact values_retained k rest store32 storage mid ka finiteK final ⟨fun _ h=>h,fun _ _=>rfl⟩
  · have qe := extension q rest store32 storage before qa
    have ke := extension k rest store32 storage mid ka
    exact ⟨fun a h=>ke.live a (qe.live a h),fun a h=>(ke.frame a (qe.live a h)).trans (qe.frame a h)⟩

#axiomsClean pair_correct
#specNonCircular Values avoiding [pair,after,typedRun,Grid.model,finalState,quantized,savedScale]
end
end VeriTile.Bench.Repairs.Int8Quantization.RealHost
