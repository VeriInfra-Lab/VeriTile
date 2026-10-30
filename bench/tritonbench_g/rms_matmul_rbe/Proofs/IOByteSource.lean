import bench.tritonbench_g.rms_matmul_rbe.Proofs.SourceExecution
import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeSourceExecution
import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOByteLoop
import bench.tritonbench_g.rms_matmul_rbe.Proofs.EpilogueComplete

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

theorem source_full {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : InputBytes fp8 p s) :
    ∃ final, run a.model (source fp8 outputFP8 p).surfaceBody s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (sourceTileValues a fp8 p s epsilon theta) s).mem ∧
      final.regs .real [] "EPS"=some (fun _ => epsilon) ∧
      final.regs .real [] "THETA"=some (fun _ => theta) ∧ final.pids=s.pids := by
  let preState := prefixExecution a.model fp8 outputFP8 p s
  have pf : preState.val.mem=s.mem := prefix_frame a.model fp8 outputFP8 p s
  have pv := prefix_values a fp8 outputFP8 p s
  obtain ⟨last,hl,hv,hf,hh,hp⟩ := IOBytes.execution a Bool.false fp8 outputFP8 p s (initialValues a p s)
    bytes (initial_regions a p s) ((p.K+63)/64) 0 preState.val (by omega) pf pv
  have ps := prefix_scalars a.model fp8 outputFP8 p s
  have pc := prefix_coordinates a.model fp8 outputFP8 p s
  have le : last.regs .real [] "EPS"=some (fun _ => epsilon) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.1.trans he)
  have lt : last.regs .real [] "THETA"=some (fun _ => theta) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.2.trans ht)
  have lb : last.regs .nat [] "pid_batch"=some (fun _ => s.pids 0) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.1
  have lm : last.regs .nat [] "pid_m"=some (fun _ => programRow p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.1
  have ln : last.regs .nat [] "pid_n"=some (fun _ => programCol p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.2
  obtain ⟨final,hepi,hmemory,eh,ep⟩ := epilogueExecutionFull a fp8 outputFP8 p last epsilon theta
    (s.pids 0) (programRow p s) (programCol p s) _ _ le lt lb lm ln hv.moments hv.accumulator
  have hstep : FP.Structural.step a.model
      (.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
        (body fp8 outputFP8 p)) preState.val=some last := by
    simpa [FP.Structural.step,evalOp_unfold,countExpr,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,IOBytes.code] using hl
  refine ⟨final,?_,?_,(eh _ _ _ (by simp [headerNames])).trans le,
    (eh _ _ _ (by simp [headerNames])).trans lt,ep.trans (hp.trans ?_)⟩
  rotate_left 2
  · cases fp8 <;> cases outputFP8 <;> rfl
  · rw [partition,List.append_assoc,run_append,preState.property]
    simp only [Option.bind_some,run_append,run,hstep,Option.bind_some]
    exact hepi
  · exact hmemory.trans (tileWrite_memory a outputFP8 p (s.pids 0)
      (programRow p s) (programCol p s) _ last s hf)

#axiomsClean source_full

theorem native_source_full {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : InputBytes fp8 p s) :
    ∃ final, run a.model (selectedSource plan fp8 outputFP8 p) s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (nativeTileValues a plan fp8 p s epsilon theta) s).mem ∧
      final.regs .real [] "EPS"=some (fun _ => epsilon) ∧
      final.regs .real [] "THETA"=some (fun _ => theta) ∧ final.pids=s.pids := by
  let preState := prefixExecution a.model fp8 outputFP8 p s
  have pf : preState.val.mem=s.mem := prefix_frame a.model fp8 outputFP8 p s
  have pv := prefix_values a fp8 outputFP8 p s
  obtain ⟨last,hl,hv,hf,hh,hp⟩ := IOBytes.execution a Bool.true fp8 outputFP8 p s (initialValues a p s)
    bytes (initial_regions a p s) ((p.K+63)/64) 0 preState.val (by omega) pf pv
  have ps := prefix_scalars a.model fp8 outputFP8 p s
  have pc := prefix_coordinates a.model fp8 outputFP8 p s
  have le : last.regs .real [] "EPS"=some (fun _ => epsilon) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.1.trans he)
  have lt : last.regs .real [] "THETA"=some (fun _ => theta) :=
    (hh _ _ _ (by simp [headerNames])).trans (ps.2.trans ht)
  have lb : last.regs .nat [] "pid_batch"=some (fun _ => s.pids 0) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.1
  have lm : last.regs .nat [] "pid_m"=some (fun _ => programRow p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.1
  have ln : last.regs .nat [] "pid_n"=some (fun _ => programCol p s) :=
    (hh _ _ _ (by simp [headerNames])).trans pc.2.2
  obtain ⟨final,hepi,hmemory,eh,ep⟩ := nativeEpilogueExecutionFull a plan fp8 outputFP8 p last epsilon theta
    (s.pids 0) (programRow p s) (programCol p s) _ _ le lt lb lm ln hv.moments hv.accumulator
  have hstep : FP.Structural.step a.model
      (.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
        (selectedBody fp8 outputFP8 p)) preState.val=some last := by
    simpa [FP.Structural.step,evalOp_unfold,countExpr,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,IOBytes.code] using hl
  refine ⟨final,?_,?_,(eh _ _ _ (by simp [headerNames])).trans le,
    (eh _ _ _ (by simp [headerNames])).trans lt,ep.trans (hp.trans ?_)⟩
  rotate_left 2
  · cases fp8 <;> cases outputFP8 <;> rfl
  · rw [selectedSource,List.append_assoc,run_append,preState.property]
    simp only [Option.bind_some,run_append,run,hstep,Option.bind_some]
    exact hepi
  · exact hmemory.trans (tileWrite_memory a outputFP8 p (s.pids 0)
      (programRow p s) (programCol p s) _ last s hf)

#axiomsClean native_source_full

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
