import bench.repairs.tritonbench_g.matmul_dequantize.Contract
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.KernelTensor
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.KernelSources
import bench.repairs.tritonbench_g.matmul_dequantize.Misc.KernelResult
import bench.repairs.tritonbench_g.matmul_dequantize.Misc.KernelRealResult
import bench.repairs.tritonbench_g.matmul_dequantize.Misc.KernelNativeResult

/-! Optional Python-check and actual Tensor input-read correspondence.
The four genuine kernel specifications remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
noncomputable section

theorem python_checks_iff_preconditions (p : Checked.PublicArguments) :
    Checked.checkArguments p = .ok () ↔ Checked.AdmittedArguments p :=
  Checked.checkArguments_iff p

theorem actual_fp32_logical_read (d : Host.Tensor) (t : TensorBinding.Tensor)
    (binding : TensorBinding.Binding d t) (dtype : d.dtype = .fp32)
    (x : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws x)
    (selected : t.fp32 = some x) (raw : ByteBridge.Memory)
    (r n : Nat) (hr : r < d.rows) (hn : n < d.columns) :
    (VeriTile.Torch.FP32.logicalValue (KernelTensor.fp32Read d raw) x laws (r*d.columns+n)).bits =
      ByteBridge.raw32 raw d (r*d.rowStride+n*d.columnStride) :=
  KernelTensor.shared_read d t binding dtype x laws selected raw r n hr hn

theorem actual_tensor_byte_read (d : Host.Tensor) (t : TensorBinding.Tensor)
    (binding : TensorBinding.Binding d t) (width : Nat) (raw : ByteBridge.Memory)
    (r n : Nat) (hr : r < d.rows) (hn : n < d.columns) (b : Nat) :
    KernelTensor.logicalByte t width raw (r*d.columns+n) b =
      raw d.region (width*(d.storageOffset+r*d.rowStride+n*d.columnStride)+b) :=
  KernelTensor.logical_byte d t binding width raw r n hr hn b

#axiomsClean python_checks_iff_preconditions
#axiomsClean actual_fp32_logical_read
#axiomsClean actual_tensor_byte_read
end
end VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge


namespace VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host Checked KernelResult
noncomputable section

/-- Complete kernel result in mathematical Real semantics, from the original
packed bytes and actual bound Tensor coordinates. No host invocation, allocator,
Torch implementation, successful execution or expected-output premise. -/
theorem gptq_kernel_correct (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq)
    (h : InputContract a .gptq) (raw : ByteBridge.Memory) (heap : RealWhole.Heap)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    Spec.Real (ABI.SourceExact a .gptq cfg ∧ ∃ final,
      Grid.run (VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.firstProgram a cfg (admitted a .gptq h)) jobs (RealWhole.entry a raw heap)=some final ∧
      RealResult a.c (t.get a.c.region) (FirstGrid.geometry (first a cfg)) Bool.true
        (RealWhole.entry a raw heap) final (RealWhole.gptqReference a cfg raw heap)) := by
  have ha := admitted a .gptq h
  have candidate : cfg∈candidates .gptq := by simp [candidates,first_exhaustive]
  obtain ⟨exact,final,run,values,frame⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.gptq a cfg ha
      (indices a .gptq cfg candidate h) (RealWhole.entry a raw heap)
      (RealWhole.packed a .gptq ha raw heap) jobs schedule
  refine ⟨exact,final,run,?_⟩
  apply real_complete a.c (t.get a.c.region) (TensorBinding.output_bound a .gptq t)
    (FirstGrid.geometry (first a cfg)) ⟨ha.cv.1,ha.cv.2.1⟩ ⟨rfl,rfl⟩ Bool.true
    (RealWhole.entry a raw heap) final (RealWhole.gptqReference a cfg raw heap) ?_ frame
  intro x
  rw [values x]
  rfl

/-- Complete kernel result in mathematical Real semantics, from the original
packed bytes and actual bound Tensor coordinates. No host invocation, allocator,
Torch implementation, successful execution or expected-output premise. -/
theorem s2_kernel_correct (a : Arguments) (t : TensorBinding.Tensors a .s2)
    (h : InputContract a .s2) (raw : ByteBridge.Memory) (heap : RealWhole.Heap)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    Spec.Real (ABI.SourceExact a .s2 .m64n256 ∧ ∃ final,
      Grid.run (VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.secondProgram a (admitted a .s2 h)) jobs (RealWhole.entry a raw heap)=some final ∧
      RealResult a.c (t.get a.c.region) (SecondGrid.geometry (second a)) Bool.true
        (RealWhole.entry a raw heap) final (RealWhole.s2Reference a raw heap)) := by
  have ha := admitted a .s2 h
  have candidate : .m64n256∈candidates .s2 := by simp [candidates]
  obtain ⟨exact,final,run,values,frame⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.s2 a ha
      (indices a .s2 .m64n256 candidate h) (RealWhole.entry a raw heap)
      (RealWhole.packed a .s2 ha raw heap) jobs schedule
  refine ⟨exact,final,run,?_⟩
  apply real_complete a.c (t.get a.c.region) (TensorBinding.output_bound a .s2 t)
    (SecondGrid.geometry (second a)) ⟨ha.cv.1,ha.cv.2.1⟩ ⟨rfl,rfl⟩ Bool.true
    (RealWhole.entry a raw heap) final (RealWhole.s2Reference a raw heap) ?_ frame
  intro x
  rw [values x]
  rfl

/-- Complete kernel result in mathematical Real semantics, from the original
packed bytes and actual bound Tensor coordinates. No host invocation, allocator,
Torch implementation, successful execution or expected-output premise. -/
theorem dequant32_kernel_correct (a : Arguments) (t : TensorBinding.Tensors a .dequant32)
    (h : InputContract a .dequant32) (raw : ByteBridge.Memory) (heap : RealWhole.Heap)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    Spec.Real (ABI.SourceExact a .dequant32 .m64n256 ∧ ∃ final,
      Grid.run (VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.dequant32Program a (admitted a .dequant32 h)) jobs (RealWhole.entry a raw heap)=some final ∧
      RealResult a.d (t.get a.d.region) (DequantGrid32.geometry (Host.dequant32 a)) Bool.false
        (RealWhole.entry a raw heap) final (RealWhole.dequant32Reference a raw heap)) := by
  have ha := admitted a .dequant32 h
  have candidate : .m64n256∈candidates .dequant32 := by simp [candidates]
  obtain ⟨exact,final,run,values,frame⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.dequant32 a ha
      (indices a .dequant32 .m64n256 candidate h) rfl (RealWhole.entry a raw heap)
      (RealWhole.packed a .dequant32 ha raw heap) jobs schedule
  refine ⟨exact,final,run,?_⟩
  apply real_complete a.d (t.get a.d.region) (TensorBinding.output_bound a .dequant32 t)
    (DequantGrid32.geometry (Host.dequant32 a)) ⟨ha.dv.1,ha.dv.2.1⟩ ⟨rfl,rfl⟩ Bool.false
    (RealWhole.entry a raw heap) final (RealWhole.dequant32Reference a raw heap) ?_ frame
  intro x
  rw [values x]
  rfl

/-- Complete kernel result in mathematical Real semantics, from the original
packed bytes and actual bound Tensor coordinates. No host invocation, allocator,
Torch implementation, successful execution or expected-output premise. -/
theorem dequant16_kernel_correct (a : Arguments) (route : Route) (route16 : route=.dequant16 ∨ route=.s1) (t : TensorBinding.Tensors a route)
    (h : InputContract a route) (raw : ByteBridge.Memory) (heap : RealWhole.Heap)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    Spec.Real (ABI.SourceExact a route .m64n256 ∧ ∃ final,
      Grid.run (VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.dequant16Program a (admitted a route h)) jobs (RealWhole.entry a raw heap)=some final ∧
      RealResult a.d (t.get a.d.region) (DequantGrid16.geometry (Host.dequant16 a)) Bool.true
        (RealWhole.entry a raw heap) final (RealWhole.dequant16Reference a raw heap)) := by
  have ha := admitted a route h
  have candidate : .m64n256∈candidates route := by rcases route16 with rfl|rfl <;> simp [candidates]
  obtain ⟨exact,final,run,values,frame⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Correct.dequant16 a ha
      (indices a route .m64n256 candidate h) route16 (RealWhole.entry a raw heap)
      (RealWhole.packed a route ha raw heap) jobs schedule
  refine ⟨exact,final,run,?_⟩
  apply real_complete a.d (t.get a.d.region) (t.bound a.d (by rcases route16 with rfl|rfl <;> simp [TensorBinding.live,Host.tensors]))
    (DequantGrid16.geometry (Host.dequant16 a)) ⟨ha.dv.1,ha.dv.2.1⟩ ⟨rfl,rfl⟩ Bool.true
    (RealWhole.entry a raw heap) final (RealWhole.dequant16Reference a raw heap) ?_ frame
  intro x
  rw [values x]
  rfl


#axiomsClean gptq_kernel_correct
#axiomsClean s2_kernel_correct
#axiomsClean dequant32_kernel_correct
#axiomsClean dequant16_kernel_correct
end
end VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge


namespace VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host Checked KernelResult
noncomputable section

/-- Successful complete source execution equals the independent source-order
word/instruction reference at every actual output Tensor byte, preserving the
entire exterior frame. One ISA is used on both sides; primitive/cast/compiler
boundaries are not a numerical experiment or a floating rewrite assumption. -/
theorem gptq_kernel_equiv (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq)
    (h : InputContract a .gptq) (plan : Native.Plan) (isa : Native.WordISA) (before : ByteBridge.Memory)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry (first a cfg))))) :
    KernelTensor.SharedReads a .gptq t before ∧ ABI.SourceExact a .gptq cfg ∧ ∃ final,
      Grid.run (FirstGrid.program Native.scalar Native.firstDot (first a cfg) (first_valid a cfg (admitted a .gptq h)) (distinct a (admitted a .gptq h)).1 (distinct a (admitted a .gptq h)).2.1 (distinct a (admitted a .gptq h)).2.2.1 (distinct a (admitted a .gptq h)).2.2.2.1) jobs (ByteBridge.entry a before)=some final ∧
      NativeResult a.c (t.get a.c.region) (FirstGrid.geometry (first a cfg)) 2 Bool.true plan isa before
        (ByteBridge.entry a before) final (NativeWhole.gptqReference a cfg before) := by
  have ha := admitted a .gptq h
  have candidate : cfg∈candidates .gptq := by simp [candidates,first_exhaustive]
  obtain ⟨exact,final,run,values,_,frame,_⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv.gptq a cfg ha
      (indices a .gptq cfg candidate h) (ByteBridge.entry a before)
      (ByteBridge.packed a .gptq ha before) (ByteBridge.input_words a .gptq ha before)
      plan isa jobs schedule
  refine ⟨KernelTensor.shared_reads a .gptq t before,exact,final,run,?_⟩
  apply native_complete a.c (t.get a.c.region) (TensorBinding.output_bound a .gptq t)
    (FirstGrid.geometry (first a cfg)) (first_valid a cfg ha) ⟨ha.cv.1,ha.cv.2.1⟩ ⟨rfl,rfl⟩
    2 (by decide) Bool.true plan isa before (ByteBridge.entry a before) final
    (NativeWhole.gptqReference a cfg before) ?_ frame
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

/-- Successful complete source execution equals the independent source-order
word/instruction reference at every actual output Tensor byte, preserving the
entire exterior frame. One ISA is used on both sides; primitive/cast/compiler
boundaries are not a numerical experiment or a floating rewrite assumption. -/
theorem s2_kernel_equiv (a : Arguments) (t : TensorBinding.Tensors a .s2)
    (h : InputContract a .s2) (plan : Native.Plan) (isa : Native.WordISA) (before : ByteBridge.Memory)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry (second a))))) :
    KernelTensor.SharedReads a .s2 t before ∧ ABI.SourceExact a .s2 .m64n256 ∧ ∃ final,
      Grid.run (SecondGrid.program Native.scalar Native.secondDot (second a) (second_valid a (admitted a .s2 h)) (distinct a (admitted a .s2 h)).1 (distinct a (admitted a .s2 h)).2.1 (distinct a (admitted a .s2 h)).2.2.1 (distinct a (admitted a .s2 h)).2.2.2.1) jobs (ByteBridge.entry a before)=some final ∧
      NativeResult a.c (t.get a.c.region) (SecondGrid.geometry (second a)) 2 Bool.true plan isa before
        (ByteBridge.entry a before) final (NativeWhole.s2Reference a before) := by
  have ha := admitted a .s2 h
  have candidate : .m64n256∈candidates .s2 := by simp [candidates]
  obtain ⟨exact,final,run,values,_,frame,_⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv.s2 a ha
      (indices a .s2 .m64n256 candidate h) (ByteBridge.entry a before)
      (ByteBridge.packed a .s2 ha before) (ByteBridge.input_words a .s2 ha before)
      plan isa jobs schedule
  refine ⟨KernelTensor.shared_reads a .s2 t before,exact,final,run,?_⟩
  apply native_complete a.c (t.get a.c.region) (TensorBinding.output_bound a .s2 t)
    (SecondGrid.geometry (second a)) (second_valid a ha) ⟨ha.cv.1,ha.cv.2.1⟩ ⟨rfl,rfl⟩
    2 (by decide) Bool.true plan isa before (ByteBridge.entry a before) final
    (NativeWhole.s2Reference a before) ?_ frame
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

/-- Successful complete source execution equals the independent source-order
word/instruction reference at every actual output Tensor byte, preserving the
entire exterior frame. One ISA is used on both sides; primitive/cast/compiler
boundaries are not a numerical experiment or a floating rewrite assumption. -/
theorem dequant32_kernel_equiv (a : Arguments) (t : TensorBinding.Tensors a .dequant32)
    (h : InputContract a .dequant32) (plan : Native.Plan) (isa : Native.WordISA) (before : ByteBridge.Memory)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a))))) :
    KernelTensor.SharedReads a .dequant32 t before ∧ ABI.SourceExact a .dequant32 .m64n256 ∧ ∃ final,
      Grid.run (DequantGrid32.program Native.scalar (Host.dequant32 a) (d32_valid a (admitted a .dequant32 h)) (distinct a (admitted a .dequant32 h)).2.2.2.2.1 (distinct a (admitted a .dequant32 h)).2.2.2.2.2.1 (distinct a (admitted a .dequant32 h)).2.2.2.2.2.2) jobs (ByteBridge.entry a before)=some final ∧
      NativeResult a.d (t.get a.d.region) (DequantGrid32.geometry (Host.dequant32 a)) 4 Bool.false plan isa before
        (ByteBridge.entry a before) final (NativeWhole.dequant32Reference a before) := by
  have ha := admitted a .dequant32 h
  have candidate : .m64n256∈candidates .dequant32 := by simp [candidates]
  obtain ⟨exact,final,run,values,_,frame,_⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv.dequant32 a ha
      (indices a .dequant32 .m64n256 candidate h) (ByteBridge.entry a before)
      (ByteBridge.packed a .dequant32 ha before) (ByteBridge.input_words a .dequant32 ha before)
      plan isa jobs schedule
  refine ⟨KernelTensor.shared_reads a .dequant32 t before,exact,final,run,?_⟩
  apply native_complete a.d (t.get a.d.region) (TensorBinding.output_bound a .dequant32 t)
    (DequantGrid32.geometry (Host.dequant32 a)) (d32_valid a ha) ⟨ha.dv.1,ha.dv.2.1⟩ ⟨rfl,rfl⟩
    4 (by decide) Bool.false plan isa before (ByteBridge.entry a before) final
    (NativeWhole.dequant32Reference a before) ?_ frame
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

/-- Successful complete source execution equals the independent source-order
word/instruction reference at every actual output Tensor byte, preserving the
entire exterior frame. One ISA is used on both sides; primitive/cast/compiler
boundaries are not a numerical experiment or a floating rewrite assumption. -/
theorem dequant16_kernel_equiv (a : Arguments) (route : Route) (route16 : route=.dequant16 ∨ route=.s1) (t : TensorBinding.Tensors a route)
    (h : InputContract a route) (plan : Native.Plan) (isa : Native.WordISA) (before : ByteBridge.Memory)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    KernelTensor.SharedReads a route t before ∧ ABI.SourceExact a route .m64n256 ∧ ∃ final,
      Grid.run (DequantGrid16.program Native.scalar (Host.dequant16 a) (d16_valid a (admitted a route h)) (distinct a (admitted a route h)).2.2.2.2.1 (distinct a (admitted a route h)).2.2.2.2.2.1 (distinct a (admitted a route h)).2.2.2.2.2.2) jobs (ByteBridge.entry a before)=some final ∧
      NativeResult a.d (t.get a.d.region) (DequantGrid16.geometry (Host.dequant16 a)) 2 Bool.true plan isa before
        (ByteBridge.entry a before) final (NativeWhole.dequant16Reference a before) := by
  have ha := admitted a route h
  have candidate : .m64n256∈candidates route := by rcases route16 with rfl|rfl <;> simp [candidates]
  obtain ⟨exact,final,run,values,_,frame,_⟩ :=
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPEquiv.dequant16 a route ha route16
      (indices a route .m64n256 candidate h) (ByteBridge.entry a before)
      (ByteBridge.packed a route ha before) (ByteBridge.input_words a route ha before)
      plan isa jobs schedule
  refine ⟨KernelTensor.shared_reads a route t before,exact,final,run,?_⟩
  apply native_complete a.d (t.get a.d.region) (t.bound a.d (by rcases route16 with rfl|rfl <;> simp [TensorBinding.live,Host.tensors]))
    (DequantGrid16.geometry (Host.dequant16 a)) (d16_valid a ha) ⟨ha.dv.1,ha.dv.2.1⟩ ⟨rfl,rfl⟩
    2 (by decide) Bool.true plan isa before (ByteBridge.entry a before) final
    (NativeWhole.dequant16Reference a before) ?_ frame
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2


#axiomsClean gptq_kernel_equiv
#axiomsClean s2_kernel_equiv
#axiomsClean dequant32_kernel_equiv
#axiomsClean dequant16_kernel_equiv
end
end VeriTile.Bench.Repairs.MatmulDequantize.TensorBridge
