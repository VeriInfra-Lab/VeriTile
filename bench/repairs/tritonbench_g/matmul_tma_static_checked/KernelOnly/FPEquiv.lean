import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.NativeTensor

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Exact bit result and actual Tensor byte frame for the recorded fixed-K16
SM90 profiles, under instruction-local reference correspondence. No native/Real
identity or sequential-FMA approximation is claimed. -/
specification kernel_fp_equiv (rest : Algebra Word) (I : Instructions) (R : Reference)
    (law : Correspondence I R) (profile : Profile) (read : Read) (h : Shape) (f : Flags) (cfg : Config)
    (_checked : InputContract h cfg) (schedule : ObservedSchedule h f cfg profile)
    (a b c : TensorABI.Tensor) (abi : Binding h f a b c) (s : State Word)
    (input : Inputs (readWord read Bool.true) h a b s) (output : Bound read f.half c (h.rows*h.columns) s) :
    ∃ typed actual bytes,
      FP.Structural.exec (backend (Native.algebra rest I) (sourceDot I profile)) (kernel h f) s=some typed ∧
      PhysicalExecution.program (Native.algebra rest I) (sourceDot I profile) h f a b c (s,read)=(actual,bytes) ∧
      typed.mem=actual.mem ∧
      Post (params h f) f.half s typed (expected (referenceInstructions R) profile (readWord read Bool.true) h f a b) ∧
      NativeTensor.TensorPost (referenceInstructions R) profile read bytes h f a b c := by
  obtain ⟨typed,actual,bytes,he,hb,hm,post,tensor⟩ := NativeTensor.correct rest I profile read h f (Addresses.observed_dimensions h f cfg profile schedule) a b c abi s input output
  have eq := instructions_eq I R law
  exact ⟨typed,actual,bytes,he,hb,hm,eq ▸ post,eq ▸ tensor⟩

#axiomsClean kernel_fp_equiv
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
