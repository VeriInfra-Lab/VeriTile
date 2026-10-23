import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.SourceReference
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeRawOutput

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

/-- Selected-native syntax execution to an independently indexed input and scalar
reference, retaining every final output cast and every untouched cell. -/
theorem native_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : Int8Memory s) (even : p.N%2=0) :
    ∃ final, run a.model (selectedSource plan fp8 outputFP8 p) s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (nativeRawTile a plan fp8 p s epsilon theta) s).mem := by
  obtain ⟨final,hex,hmem⟩ := nativeSourceExecution a plan fp8 outputFP8 p s epsilon theta he ht bytes
  refine ⟨final,hex,hmem.trans (congrArg State.mem (tileWrite_congr a outputFP8 p
    (s.pids 0) (programRow p s) (programCol p s) _ _ s ?_))⟩
  intro i hi
  rcases i with ⟨row,lane,⟨⟩⟩
  exact nativeTile_reference a z plan fp8 p s epsilon theta even row lane hi.1 hi.2

#axiomsClean native_reference
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
