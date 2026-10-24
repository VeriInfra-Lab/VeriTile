import bench.repairs.tritonbench_g.int8_quantization.Proofs.DirectResults
namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option maxHeartbeats 400000

/-- Complete observable values, including the zero-channel scale branch.
Source precision is retained by the supplied operation-level backend. -/
structure Values (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (initial : Memory) (l : Bytes.Layout) (output scales : Tensor) (result : Memory) : Prop where
  outputShape : output.metadata.shape=c.x.metadata.shape
  scaleShape : scales.metadata.shape=Direct.scaleShape c.x.metadata c.block
  outputType : output.metadata.dtype=.other "int8"
  scaleType : scales.metadata.dtype=.floating .fp32
  outputDevice : output.metadata.device=c.x.metadata.device
  scaleDevice : scales.metadata.device=c.x.metadata.device
  bytes : ∀ w, w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c), Grid.active (p c) w i →
    result (output.base+Grid.offset (p c) w i)=
      BitVec.ofInt 8 (quantized M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
        (Grid.atProgram (Bytes.bind l initial) w) i)
  scales : ∀ w, w∈Schedule.programs 0 (count c) →
    Bytes.read32 result (scales.base+4*w.index)=
      if channels c.x.metadata=0 then M.literal (some .fp32) .real 0 else
      savedScale M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
        (Grid.atProgram (Bytes.bind l initial) w)

theorem values (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (out : Allocation c before) :
    Values c M ops before.memory (layout c out) out.output out.scales (after c M ops before out).memory :=
  ⟨out.outputShape,out.scaleShape,out.outputType,out.scaleType,out.outputDevice,out.scaleDevice,
    output_bytes c M ops before out,scale_bytes c M ops before out⟩

theorem values_retained (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (out : Allocation c before) (final : Heap)
    (ext : Extension (after c M ops before out) final) :
    Values c M ops before.memory (layout c out) out.output out.scales final.memory := by
  refine ⟨out.outputShape,out.scaleShape,out.outputType,out.scaleType,out.outputDevice,out.scaleDevice,?_,?_⟩
  · intro w hw i hi
    rw [ext.frame _ ?_]
    · exact output_bytes c M ops before out w hw i hi
    · apply Or.inr;apply Or.inl
      have bound := offset_bound c w hw i hi
      rw [numel_eq _ _ c.contract] at bound
      change out.output.base ≤ out.output.base+Grid.offset (p c) w i ∧
        out.output.base+Grid.offset (p c) w i < out.output.base+(p c).batches*(p c).length*(p c).channels
      exact ⟨Nat.le_add_right _ _,Nat.add_lt_add_left bound _⟩
  · intro w hw
    rw [←scale_bytes c M ops before out w hw]
    funext b
    apply ext.frame
    apply Or.inr;apply Or.inr
    have bound : w.index < count c := by simpa using (Schedule.program_bounds 0 (count c) w hw).2.2
    have hb := b.isLt
    change out.scales.base ≤ out.scales.base+4*w.index+b.val ∧
      out.scales.base+4*w.index+b.val < out.scales.base+4*count c
    omega

theorem pair_correct (q k : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q M ops before qa)) :
    ∃ result, pair q k M ops before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      Values q M ops before.memory (layout q qa) result.q result.qScale result.heap.memory ∧
      Values k M ops (after q M ops before qa).memory (layout k ka) result.k result.kScale result.heap.memory ∧
      Extension before result.heap := by
  let mid := after q M ops before qa
  let final := after k M ops mid ka
  refine ⟨⟨qa.output,qa.scales,ka.output,ka.scales,final⟩,?_,rfl,rfl,rfl,rfl,?_,?_,?_⟩
  · simp [pair,(checkTensorInputs_iff _ _).mpr q.contract,(checkTensorInputs_iff _ _).mpr k.contract,mid,final,Except.bind,bind,pure,Except.pure]
  · exact values_retained q M ops before qa final (extension k M ops mid ka)
  · exact values k M ops mid ka
  · have qext := extension q M ops before qa
    have kext := extension k M ops mid ka
    refine ⟨fun a h=>kext.live a (qext.live a h),fun a h=>?_⟩
    exact (kext.frame a (qext.live a h)).trans (qext.frame a h)

#axiomsClean pair_correct
#axiomsClean values_retained
end
end VeriTile.Bench.Repairs.Int8Quantization.Direct
