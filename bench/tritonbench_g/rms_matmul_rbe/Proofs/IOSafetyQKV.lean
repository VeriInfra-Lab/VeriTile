import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOSafetySource
import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOOutput

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem bounds_pids (p : Arguments) (s t : State α) (same : t.pids=s.pids)
    (capacity : RegionName → Nat) (bound : Bounds p s capacity) : Bounds p t capacity := by
  rcases bound with ⟨hx,hw,hr,ho⟩
  constructor
  · simpa only [programRow,same] using hx
  · simpa only [programCol,same] using hw
  · exact hr
  · simpa only [programRow,programCol,same] using ho

theorem qkv_safe (a : Arithmetic α) (z : ZeroLaws a) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _=>epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _=>theta))
    (bytes : ∀ phase,IOBytes.InputBytes fp8 (qkvArgument p phase) s)
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (capacity : RegionName → Nat) (bound : ∀ phase,Bounds (qkvArgument p phase) s capacity) :
    traceSafeList a.model capacity (qkvCode native fp8 p) s := by
  let q := qkvArgument p 0
  let k := qkvArgument p 1
  let v := qkvArgument p 2
  obtain ⟨qs,hq,mq,eq,tq,pq⟩ := IOBytes.callee_complete a z native (selectedRotaryPlan 0)
    fp8 Bool.false q s epsilon theta he ht (bytes 0) (even 0)
  have qi (reader : Fin 3) : InputMemory (qkvArgument p reader) qs s := by
    simpa only [InputMemory,mq] using calleeWrite_inputs a native (selectedRotaryPlan 0)
      fp8 Bool.false q (qkvArgument p reader) s s epsilon theta (apart 0 reader)
  have qb : IOBytes.InputBytes fp8 k qs := IOBytes.inputBytes_stable fp8 k s qs (qi 1) (bytes 1)
  obtain ⟨ks,hk,mk,ek,tk,pk⟩ := IOBytes.callee_complete a z native (selectedRotaryPlan 1)
    fp8 fp8 k qs epsilon theta eq tq qb (even 1)
  have ki (reader : Fin 3) : InputMemory (qkvArgument p reader) ks s := by
    have current : InputMemory (qkvArgument p reader) ks qs := by
      simpa only [InputMemory,mk] using calleeWrite_inputs a native (selectedRotaryPlan 1)
        fp8 fp8 k (qkvArgument p reader) qs qs epsilon theta (apart 1 reader)
    exact InputMemory.trans _ ks qs s current (qi reader)
  have kp : ks.pids=s.pids := pk.trans pq
  have kb : IOBytes.InputBytes fp8 v ks := IOBytes.inputBytes_stable fp8 v s ks (ki 2) (bytes 2)
  have sq:=callee_safe a native (selectedRotaryPlan 0) fp8 Bool.false q s epsilon theta he ht
    (bytes 0) capacity (bound 0)
  have sk:=callee_safe a native (selectedRotaryPlan 1) fp8 fp8 k qs epsilon theta eq tq qb
    capacity (bounds_pids k s qs pq capacity (bound 1))
  have sv:=callee_safe a native (selectedRotaryPlan 2) fp8 fp8 v ks epsilon theta ek tk kb
    capacity (bounds_pids v s ks kp capacity (bound 2))
  rw [qkvCode,List.append_assoc,traceSafeList_append]
  refine ⟨sq,?_⟩
  intro qstate qr
  have equal : qstate=qs := Option.some.inj (qr.symm.trans hq)
  subst qstate
  rw [traceSafeList_append]
  refine ⟨sk,?_⟩
  intro kstate kr
  have equal : kstate=ks := Option.some.inj (kr.symm.trans hk)
  subst kstate
  exact sv

#axiomsClean qkv_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
