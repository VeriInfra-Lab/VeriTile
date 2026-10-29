import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardPrologue
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def forwardScoreResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) : State α :=
  let t := current.setReg "start_n" .nat [] (fun _ => c*p.blockN)
  let t := t.setReg "k" p.dtype.toTileDType [p.blockN,p.head] (forwardKV p initial p.k c)
  let t := t.setReg "qk" .real [p.blockM,p.blockN] (fun _ => forwardZero P)
  let t := t.setReg "qk" .real [p.blockM,p.blockN] (fun i =>
    P.binary (some .fp32) .real .add (forwardZero P) (P.cast (some .fp32) .fp32 .real (forwardDot P p initial c i)))
  let t := t.setReg "qk" .real [p.blockM,p.blockN] (fun i =>
    P.binary (some .fp32) .real .mul
      (P.binary (some .fp32) .real .add (forwardZero P) (P.cast (some .fp32) .fp32 .real (forwardDot P p initial c i)))
      (P.literal (some .fp32) .real p.scale))
  t.setReg "qk" .real [p.blockM,p.blockN] (forwardScore P p initial c)

theorem forwardScoreExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (h : ForwardInvariant P p initial c st current) :
    run (backend rest P) ((forwardBody p).take 5)
      (current.setReg "start_n" .nat [] (fun _ => c*p.blockN))=
      some (forwardScoreResult P p initial current c) := by
  simp [forwardBody,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,numericLe,backend,forwardScoreResult,
    h.query,h.keyPtr,h.memory,h.undef,h.rows,h.cols,forwardKVPointer,
    forwardZero,forwardDot,forwardScore,forwardKV,forwardKVOffset,
    Region.cast,FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,TileShape.indexToList]
  delta forwardKV forwardScore forwardDot forwardKVOffset FP.Structural.bop
  simp [natLe,Option.bind,decide_eq_true_eq,Broadcast.leftIndex,Broadcast.rightIndex,forwardKVPointer,TileShape.indexToList,forwardZero,transposeValues]

theorem forwardScoreInvariant (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) (h : ForwardInvariant P p initial c st current) :
    ForwardInvariant P p initial c st (forwardScoreResult P p initial current c) := by
  constructor <;> simp [forwardScoreResult,h.memory,h.undef,h.pids,h.start,h.hz,h.rows,h.cols,
    h.query,h.keyPtr,h.valuePtr,h.outputPtr,h.maximum,h.denominator,h.numerator]

#axiomsClean forwardScoreExecution
#axiomsClean forwardScoreInvariant
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
