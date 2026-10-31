import bench.tritonbench_g.rms_matmul_rbe.Proofs.Callee

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000

def qkvArgument (p : Fin 3 → Arguments) (phase : Fin 3) : Arguments :=
  {p phase with rotary:=decide (phase.val<2)}

def qkvOutputFP8 (fp8 : Bool) (phase : Fin 3) := if phase.val=0 then Bool.false else fp8

def qkvCode (native fp8 : Bool) (p : Fin 3 → Arguments) :=
  calleeCode native (selectedRotaryPlan 0) fp8 Bool.false (qkvArgument p 0) ++
  calleeCode native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1) ++
  calleeCode native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2)

def qkvWrite {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (initial : State α) (epsilon theta : α) : State α :=
  let q := calleeWrite a native (selectedRotaryPlan 0) fp8 Bool.false
    (qkvArgument p 0) initial initial epsilon theta
  let k := calleeWrite a native (selectedRotaryPlan 1) fp8 fp8
    (qkvArgument p 1) initial q epsilon theta
  calleeWrite a native (selectedRotaryPlan 2) fp8 fp8
    (qkvArgument p 2) initial k epsilon theta

/-- Literal Q→K→V composition. All three independent targets use the same
initial input snapshot and the same scalar words. Destinations remain
ordinary existing memory throughout the sequence. -/
theorem qkv_complete {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (native fp8 : Bool) (p : Fin 3 → Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta)) (bytes : Int8Memory s)
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
    fp8 Bool.false q s epsilon theta he ht bytes (even 0)
  have qb : Int8Memory qs := by
    intro r o
    rw [mq]
    exact calleeWrite_int8 a native (selectedRotaryPlan 0) fp8 Bool.false q s s epsilon theta bytes r o
  have qi (reader : Fin 3) : InputMemory (qkvArgument p reader) qs s := by
    simpa only [InputMemory,mq] using calleeWrite_inputs a native (selectedRotaryPlan 0)
      fp8 Bool.false q (qkvArgument p reader) s s epsilon theta (apart 0 reader)
  obtain ⟨ks,hk,mk,ek,tk,pk⟩ := callee_complete a z native (selectedRotaryPlan 1)
    fp8 fp8 k qs epsilon theta eq tq qb (even 1)
  have kb : Int8Memory ks := by
    intro r o
    rw [mk]
    exact calleeWrite_int8 a native (selectedRotaryPlan 1) fp8 fp8 k qs qs epsilon theta qb r o
  have ki (reader : Fin 3) : InputMemory (qkvArgument p reader) ks s := by
    have current : InputMemory (qkvArgument p reader) ks qs := by
      simpa only [InputMemory,mk] using calleeWrite_inputs a native (selectedRotaryPlan 1)
        fp8 fp8 k (qkvArgument p reader) qs qs epsilon theta (apart 1 reader)
    exact InputMemory.trans _ ks qs s current (qi reader)
  have kp : ks.pids=s.pids := pk.trans pq
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
#specNonCircular qkvWrite avoiding [qkvCode,qkv_complete,run,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
