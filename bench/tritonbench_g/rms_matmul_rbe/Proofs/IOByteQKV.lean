import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOByteCallee
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKV
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
/-- Literal Q→K→V composition. All three independent targets use the same
initial input snapshot and the same scalar words. Destinations remain
ordinary existing memory throughout the sequence. -/
theorem qkv_complete {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (native fp8 : Bool) (p : Fin 3 → Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta)) (bytes : ∀ phase, InputBytes fp8 (qkvArgument p phase) s)
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    ∃ final,run a.model (qkvCode native fp8 p) s=some final ∧
      final.mem=(qkvWrite a native fp8 p s epsilon theta).mem ∧
      final.regs .real [] "EPS"=some (fun _ => epsilon) ∧
      final.regs .real [] "THETA"=some (fun _ => theta) ∧ final.pids=s.pids := by
  let q := qkvArgument p 0
  let k := qkvArgument p 1
  let v := qkvArgument p 2
  obtain ⟨qs,hq,mq,eq,tq,pq⟩ := callee_complete a z native (selectedRotaryPlan 0)
    fp8 Bool.false q s epsilon theta he ht (bytes 0) (even 0)
  have qi (reader : Fin 3) : InputMemory (qkvArgument p reader) qs s := by
    simpa only [InputMemory,mq] using calleeWrite_inputs a native (selectedRotaryPlan 0)
      fp8 Bool.false q (qkvArgument p reader) s s epsilon theta (apart 0 reader)
  have qb : InputBytes fp8 k qs := inputBytes_stable fp8 k s qs (qi 1) (bytes 1)
  obtain ⟨ks,hk,mk,ek,tk,pk⟩ := callee_complete a z native (selectedRotaryPlan 1)
    fp8 fp8 k qs epsilon theta eq tq qb (even 1)
  have ki (reader : Fin 3) : InputMemory (qkvArgument p reader) ks s := by
    have current : InputMemory (qkvArgument p reader) ks qs := by
      simpa only [InputMemory,mk] using calleeWrite_inputs a native (selectedRotaryPlan 1)
        fp8 fp8 k (qkvArgument p reader) qs qs epsilon theta (apart 1 reader)
    exact InputMemory.trans _ ks qs s current (qi reader)
  have kp : ks.pids=s.pids := pk.trans pq
  have kb : InputBytes fp8 v ks := inputBytes_stable fp8 v s ks (ki 2) (bytes 2)
  obtain ⟨vs,hv,mv,ev,tv,pv⟩ := callee_complete a z native (selectedRotaryPlan 2)
    fp8 fp8 v ks epsilon theta ek tk kb (even 2)
  let qw := calleeWrite a native (selectedRotaryPlan 0) fp8 Bool.false q s s epsilon theta
  let kw := calleeWrite a native (selectedRotaryPlan 1) fp8 fp8 k s qw epsilon theta
  have km : ks.mem=kw.mem := by
    exact mk.trans ((congrArg State.mem (calleeWrite_initial a native (selectedRotaryPlan 1)
      fp8 fp8 k s qs epsilon theta (qi 1) pq)).trans
      (calleeWrite_memory a native (selectedRotaryPlan 1) fp8 fp8 k s qs qw epsilon theta mq))
  refine ⟨vs,?_,?_,ev,tv,pv.trans kp⟩
  · rw [qkvCode,List.append_assoc,run_append,hq]
    simp only [Option.bind_some,run_append]
    rw [hk]
    exact hv
  · exact mv.trans ((congrArg State.mem (calleeWrite_initial a native (selectedRotaryPlan 2)
      fp8 fp8 v s ks epsilon theta (ki 2) kp)).trans
      (calleeWrite_memory a native (selectedRotaryPlan 2) fp8 fp8 v s ks kw epsilon theta km))

#axiomsClean qkv_complete
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
