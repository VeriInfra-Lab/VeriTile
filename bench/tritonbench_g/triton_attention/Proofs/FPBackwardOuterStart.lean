import bench.tritonbench_g.triton_attention.Proofs.FPBackwardPrologue
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def backwardOuterStartResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c : Nat) (mem : Memory α) : State α :=
  let t := current.setReg "start_n" .nat [] (fun _ => c)
  let t := t.setReg "lo" .nat [] (fun _ => c*p.blockM)
  let t := t.setReg "offs_qm" .nat [p.blockM] (fun i => c*p.blockM+i.1.val)
  let t := t.setReg "offs_n" .nat [p.blockM] (fun i => c*p.blockM+i.1.val)
  let t := t.setReg "offs_m" .nat [p.blockM] (fun i => i.1.val)
  let t := t.setReg "offs_k" .nat [p.head] (fun i => i.1.val)
  let t := t.setReg "dq_ptrs" .ptr [p.blockM,p.head] (fun i =>
    (p.dq,backwardByteBase p initial+(c*p.blockM+i.1.val)*p.head+i.2.1.val))
  let t := t.setReg "D_ptrs" .ptr [] (fun _ => (p.delta,initial.pids 0*(p.blockM*p.blocks)))
  let t := t.setReg "m_ptrs" .ptr [] (fun _ => (p.maximum,initial.pids 0*(p.blockM*p.blocks)))
  let t := t.setReg "dv" .real [p.blockM,p.head] (backwardInnerInitial P p mem).dv
  let t := t.setReg "dk" .real [p.blockM,p.head] (backwardInnerInitial P p mem).dk
  let t := t.setReg "k" p.dtype.toTileDType [p.blockM,p.head] (backwardRead p initial mem p.k c)
  t.setReg "v" p.dtype.toTileDType [p.blockM,p.head] (backwardRead p initial mem p.v c)

theorem backwardOuterStartExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (mem : Memory α) (h : BackwardEntryInvariant p initial c mem current) :
    run (backend rest P) ((backwardOuterBody p).take 12) (current.setReg "start_n" .nat [] (fun _ => c))=
      some (backwardOuterStartResult P p initial current c mem) := by
  simp [backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,evalOffsets,backend,backwardOuterStartResult,
    h.hz,h.z,h.h,h.strideZ,h.strideH,h.dqBase,h.keyPtr,h.valuePtr,h.memory,h.undef,
    backwardPointer,backwardHeadRow,backwardByteBase,backwardInnerInitial,backwardRead,backwardOffset,
    TileShape.indexToList,Region.cast,Nat.add_assoc,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta backwardRead backwardOffset FP.Structural.bop
  simp [backwardPointer,backwardHeadRow,TileShape.indexToList,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]
  all_goals rfl

theorem backwardOuterStartInvariant (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (mem : Memory α) (h : BackwardEntryInvariant p initial c mem current) :
    BackwardInnerInvariant p initial c c mem (backwardInnerInitial P p mem)
      (backwardOuterStartResult P p initial current c mem) := by
  constructor
  · constructor
    · constructor <;> simp [backwardOuterStartResult,h.undef,h.pids,h.hz,h.z,h.h,h.strideZ,h.strideH,h.dqBase]
    all_goals simp [backwardOuterStartResult,backwardInnerInitial,h.memory,h.keyPtr,h.valuePtr,h.dkPtr,h.dvPtr]
  all_goals simp [backwardOuterStartResult,h.qPtr,h.doPtr,h.dqPtr]

#axiomsClean backwardOuterStartExecution
#axiomsClean backwardOuterStartInvariant
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
