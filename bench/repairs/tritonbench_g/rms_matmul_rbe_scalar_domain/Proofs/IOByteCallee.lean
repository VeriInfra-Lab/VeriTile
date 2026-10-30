import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOByteSource
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Callee

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem callee_complete (a : Arithmetic α) (z : ZeroLaws a)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _=>epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _=>theta))
    (bytes : InputBytes fp8 p s) (even : p.N%2=0) :
    ∃ final,run a.model (calleeCode native plan fp8 outputFP8 p) s=some final ∧
      final.mem=(calleeWrite a native plan fp8 outputFP8 p s s epsilon theta).mem ∧
      final.regs .real [] "EPS"=some (fun _=>epsilon) ∧
      final.regs .real [] "THETA"=some (fun _=>theta) ∧ final.pids=s.pids := by
  cases native
  · obtain ⟨final,hx,hm,he,ht,hp⟩:=source_full a fp8 outputFP8 p s epsilon theta he ht bytes
    refine ⟨final,hx,hm.trans (congrArg State.mem (tileWrite_congr a outputFP8 p
      (s.pids 0) (programRow p s) (programCol p s) _ _ s ?_)),he,ht,hp⟩
    intro i hi
    rcases i with ⟨row,lane,⟨⟩⟩
    exact sourceTile_reference a z fp8 p s epsilon theta even row lane hi.1 hi.2
  · obtain ⟨final,hx,hm,he,ht,hp⟩:=native_source_full a plan fp8 outputFP8 p s epsilon theta he ht bytes
    refine ⟨final,hx,hm.trans (congrArg State.mem (tileWrite_congr a outputFP8 p
      (s.pids 0) (programRow p s) (programCol p s) _ _ s ?_)),he,ht,hp⟩
    intro i hi
    rcases i with ⟨row,lane,⟨⟩⟩
    exact nativeTile_reference a z plan fp8 p s epsilon theta even row lane hi.1 hi.2

theorem inputBytes_stable (fp8 : Bool) (p : Arguments) (s t : State α)
    (same : InputMemory p t s) (bytes : InputBytes fp8 p s) : InputBytes fp8 p t := by
  intro fp r member off
  rcases member with rfl|rfl
  · rw [same.2.1]; exact bytes fp _ (Or.inl rfl) off
  · rw [same.2.2]; exact bytes fp _ (Or.inr rfl) off

#axiomsClean callee_complete
#axiomsClean inputBytes_stable
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
