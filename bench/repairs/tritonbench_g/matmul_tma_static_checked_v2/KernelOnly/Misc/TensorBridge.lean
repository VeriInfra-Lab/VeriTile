import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceModels
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Correct
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.matmul_tma_static_checked.KernelOnly.Correct
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.NativeTensor

/-! Optional recorded-profile application of the complete native kernel result. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.TensorBridge
open VeriTile.Triton FP.Structural FPValue Host
open StaticChecked.TensorABI StaticChecked.Native StaticChecked.Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem kernel_real (rest : Algebra ℝ) (read : Nat → ℝ) (h : Shape) (f : Flags) (cfg : Config)
    (checked : InputContract h cfg) (a b c : StaticChecked.TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State ℝ) (input : Inputs read h a b s) :
    VeriTile.Spec.Real (∃ t, exec (backend (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot) (kernel h f) s=some t ∧
      Post (params h f) f.half s t (StaticChecked.RealModel.product read h f a b) ∧
      StaticChecked.RealModel.TensorPost read (StaticChecked.RealModel.render read h f c t) h f a b c) := by
  obtain ⟨t,ran,post,tensor⟩ := StaticChecked.kernel_real rest read h f cfg
    checked.toInputContract a b c abi s input
  refine ⟨t,?_,post,tensor⟩
  exact (SourceModels.real_execution rest (params h f) f.half s).trans ran

/-- Complete checked-domain generic native theorem. The fixed K16 profile is
an explicit instruction-local realization boundary. Compilation samples are
not an input-domain restriction; source dimensions are derived from guards. -/
theorem kernel_fp_equiv (rest : Algebra Word) (I : Instructions) (R : Reference)
    (law : Correspondence I R) (profile : Profile) (read : Read) (h : Shape) (f : Flags) (cfg : Config)
    (checked : InputContract h cfg) (a b c : StaticChecked.TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State Word) (input : Inputs (readWord read Bool.true) h a b s)
    (output : Bound read f.half c (h.rows*h.columns) s) :
    ∃ typed actual bytes,
      FP.Structural.exec (backend (SourceNative.algebra rest I) (sourceDot I profile)) (kernel h f) s=some typed ∧
      StaticChecked.PhysicalExecution.program (StaticChecked.Native.algebra rest I) (sourceDot I profile) h f a b c (s,read)=(actual,bytes) ∧
      typed.mem=actual.mem ∧
      Post (params h f) f.half s typed (expected (referenceInstructions R) profile (readWord read Bool.true) h f a b) ∧
      StaticChecked.NativeTensor.TensorPost (referenceInstructions R) profile read bytes h f a b c := by
  obtain ⟨typed,actual,bytes,he,hb,hm,post,tensor⟩ := StaticChecked.NativeTensor.correct rest I profile read h f
    (dimensions h cfg checked) a b c abi s input output
  have eq := instructions_eq I R law
  refine ⟨typed,actual,bytes,?_,hb,hm,eq ▸ post,eq ▸ tensor⟩
  exact (SourceModels.native_execution rest I profile (params h f) f.half s).trans he

#axiomsClean kernel_real
#axiomsClean kernel_fp_equiv

/-- The twelve recorded compiler profiles are a separate evidence corollary. -/
theorem observed_profile (rest : Algebra Word) (I : Instructions) (R : Reference)
    (law : Correspondence I R) (profile : Profile) (read : Read) (h : Shape) (f : Flags) (cfg : Config)
    (checked : InputContract h cfg) (_observed : ObservedSchedule h f cfg profile)
    (a b c : StaticChecked.TensorABI.Tensor) (abi : Binding h f a b c) (s : State Word)
    (input : Inputs (readWord read Bool.true) h a b s) (output : Bound read f.half c (h.rows*h.columns) s) :
    ∃ typed actual bytes,
      FP.Structural.exec (backend (SourceNative.algebra rest I) (sourceDot I profile)) (kernel h f) s=some typed ∧
      StaticChecked.PhysicalExecution.program (StaticChecked.Native.algebra rest I) (sourceDot I profile) h f a b c (s,read)=(actual,bytes) ∧
      typed.mem=actual.mem ∧
      Post (params h f) f.half s typed (expected (referenceInstructions R) profile (readWord read Bool.true) h f a b) ∧
      StaticChecked.NativeTensor.TensorPost (referenceInstructions R) profile read bytes h f a b c :=
  kernel_fp_equiv rest I R law profile read h f cfg checked a b c abi s input output

#axiomsClean observed_profile
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.TensorBridge
