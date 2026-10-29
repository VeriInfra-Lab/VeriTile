import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.IOResults
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.NativeInputBinding

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface.Result
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

/-- The old compiler-bound native result is a source-to-half-word observation.
All raw int8/half patterns and the original scalar/K16 instruction parameters
remain admitted. No source-versus-native floating equality is asserted. -/
theorem native (a : Words.Instructions) (p : Parameters) :
    (programIO Bool.true (Words.arithmetic a) (Words.gates a) p).ObservesOn
      (fun _ _ => Words.half) byteDomain (Reference.nativeFormula a p) := by
  intro xs hd initial hin
  have hr := input_reads Bool.true (Words.arithmetic a) (Words.gates a) p xs initial hin
  have hb := hr.bytes hd
  obtain ⟨final,he,hv,hf⟩ := Grid.whole Bool.true (Words.arithmetic a) (Words.gates a) p initial hb
  refine ⟨fun bounds bounded => program_safe Bool.true (Words.arithmetic a) (Words.gates a)
    p xs bounds bounded initial hb,final,?_,?_,output_frame _ _ _ _ _ _ _ hf⟩
  · rw [program_run]; exact he
  · intro o j _
    have ho : o=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance o 0
    subst o
    change mapCell Words.half (mapCell id (final.mem "OUT" (p.columns*j.1.val+j.2.1.val))) =
      Cell.mk .fp16 (Words.half (Reference.nativeValue (Words.arithmetic a) (Words.gates a)
        p xs j.1.val j.2.1.val))
    simp only [Grid.address] at hv
    rw [hv (j.1,j.2.1) trivial]
    simp only [Grid.target,ite_true,NativeReference.logical_eq,mapCell,mapValue,id_eq]
    exact congrArg (fun x => Cell.mk .fp16 (Words.half x))
      (hr.native_target (Words.arithmetic a) (Words.gates a) p j.1.val j.2.1.val)

#axiomsClean native
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface.Result
