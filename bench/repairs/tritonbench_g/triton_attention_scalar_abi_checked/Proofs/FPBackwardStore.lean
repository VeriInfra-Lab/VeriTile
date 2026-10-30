import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardGradient
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def backwardStoreRegisters (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p) : State α :=
  let key := backwardRead p initial outerMem p.k c
  let value := backwardRead p initial outerMem p.v c
  let t := current.setReg "dk" .real [p.blockM,p.head] (backwardInnerAdvance P p initial key value c j st).dk
  let t := t.setReg "dq" .real [p.blockM,p.head] (fun i =>
    P.cast (some .fp32) .fp32 .real ((st.mem p.dq (backwardOffset p initial j i)).read .fp32))
  t.setReg "dq" .real [p.blockM,p.head] (backwardQueryValue P p initial st.mem key value c j)
def backwardStoreResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p) : State α :=
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i => t.write p.dq (backwardOffset p initial j i)
    (backwardQueryCell P p initial st.mem (backwardRead p initial outerMem p.k c)
      (backwardRead p initial outerMem p.v c) c j i)) (backwardStoreRegisters P p initial current c j outerMem st)

theorem backwardStoreExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (hm : current.mem=st.mem)
    (hptr : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _ => backwardPointer p initial p.dq j))
    (hk : current.regs p.dtype.toTileDType [p.blockM,p.head] "k"=some (backwardRead p initial outerMem p.k c))
    (hq : current.regs p.dtype.toTileDType [p.blockM,p.head] "q"=some (backwardRead p initial st.mem p.q j))
    (hdk : current.regs .real [p.blockM,p.head] "dk"=some st.dk)
    (hds : current.regs .real [p.blockM,p.blockM] "ds"=some
      (backwardDS P p initial st.mem (backwardRead p initial outerMem p.k c) (backwardRead p initial outerMem p.v c) c j)) :
    run (backend rest P) (((backwardInnerBody p).drop 12).take 4) current=
      some (backwardStoreResult P p initial current c j outerMem st) := by
  simp only [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,List.drop,List.take]
  try simp only [eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq,id_eq]
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,numeric,store,backend,backwardStoreResult,backwardStoreRegisters,
    hm,hptr,hk,hq,hdk,hds,backwardPointer,backwardOffset,
    TileShape.indexToList,Region.cast,Nat.add_assoc,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta backwardInnerAdvance backwardKeyDot backwardQueryDot backwardQueryValue backwardQueryCell backwardOffset FP.Structural.bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc,transposeValues]
  all_goals rfl

def backwardAdvancePointers (p : BackwardParameters) (initial current : State α) (j : Nat) : State α :=
  let t := current.setReg "dq_ptrs" .ptr [p.blockM,p.head] (fun i =>
    (p.dq,backwardByteBase p initial+((j+1)*p.blockM+i.1.val)*p.head+i.2.1.val))
  let t := t.setReg "q_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.q (j+1))
  let t := t.setReg "do_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.gradient (j+1))
  t.setReg "dq_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.dq (j+1))

theorem backwardAdvanceExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (j : Nat)
    (hdp : current.regs .ptr [p.blockM,p.head] "dq_ptrs"=some (fun i =>
      (p.dq,backwardByteBase p initial+(j*p.blockM+i.1.val)*p.head+i.2.1.val)))
    (hq : current.regs .blockPtr [p.blockM,p.head] "q_tile_ptr"=some (fun _ => backwardPointer p initial p.q j))
    (hdo : current.regs .blockPtr [p.blockM,p.head] "do_tile_ptr"=some (fun _ => backwardPointer p initial p.gradient j))
    (hdq : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _ => backwardPointer p initial p.dq j)) :
    run (backend rest P) ((backwardInnerBody p).drop 16) current=some (backwardAdvancePointers p initial current j) := by
  simp only [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,List.drop,List.take]
  try simp only [eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq,id_eq]
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,numeric,backend,backwardAdvancePointers,
    hdp,hq,hdo,hdq,backwardPointer,Nat.add_mul,Nat.add_assoc,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
  simp only [Nat.add_mul,Nat.one_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  congr 6
  funext i
  apply Prod.ext
  · rfl
  · omega

#axiomsClean backwardStoreExecution
#axiomsClean backwardAdvanceExecution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
