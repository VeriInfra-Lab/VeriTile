import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardLoop
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
@[simp] theorem maskedScatter_regs {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Prop) [DecidablePred mask] (s : State α) :
    (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s).regs=s.regs := by
  rw [State.masked_scatter_eq_filter,State.scatter_regs]
@[simp] theorem maskedScatter_undef {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Prop) [DecidablePred mask] (s : State α) :
    (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s).undef=s.undef := by
  rw [State.masked_scatter_eq_filter,scatterUndef]
theorem maskedScatterMemoryReference {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Prop) [DecidablePred mask] (s : State α) :
    (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s).mem=
      l.foldl (fun m i => if mask i then writeMemory m r (off i) (v i) else m) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    simp only [List.foldl_cons]
    by_cases ha : mask i
    · simp only [ha,ite_true,ih]
      rfl
    · simp only [ha,ite_false,ih]

def backwardOutputCell (P : Primitives α) (p : BackwardParameters) (v : BLane p → α) (i : BLane p) : Cell α :=
  .mk p.dtype.toTileDType (ofFloat p.dtype (P.cast none .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (v i))))
theorem backwardPointer_reset (p : BackwardParameters) (s : State α) (r : RegionName) (c : Nat) :
    (backwardPointer p s r p.blocks).advance
      [(c*p.blockM : Nat) + (1 - (p.blocks : Int))*p.blockM,0] =
      backwardPointer p s r (c+1) := by
  simp only [backwardPointer,BlockPtr.advance_2d_offsets_int]
  have eqn : ((backwardHeadRow p s+p.blocks*p.blockM : Nat) : Int) +
      ((c*p.blockM : Nat) + (1-(p.blocks : Int))*p.blockM) =
      ((backwardHeadRow p s+(c+1)*p.blockM : Nat) : Int) := by
    push_cast
    ring
  rw [eqn]
  simp only [Int.toNat_natCast]
  all_goals rfl

@[simp] theorem reset_offset (a b c m : Nat) :
    (((a+b*m : Nat) : Int) + ((c*m : Nat) + (1-(b : Int))*m)).toNat = a+(c+1)*m := by
  have he : ((a+b*m : Nat) : Int) + ((c*m : Nat) + (1-(b : Int))*m) =
      ((a+(c+1)*m : Nat) : Int) := by push_cast; ring
  rw [he]
  exact Int.toNat_natCast _

def backwardFinishResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c : Nat) (st : BackwardInner α p) : State α :=
  let t := current.setReg "q_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.q (c+1))
  let t := t.setReg "do_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.gradient (c+1))
  let t := t.setReg "dq_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.dq (c+1))
  let t := t.setReg "k_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.k (c+1))
  let t := t.setReg "v_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.v (c+1))
  let t := (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    if (backwardPointer p initial p.dv c).inBounds (TileShape.indexToList _ i) [0,1] then
      t.write p.dv (backwardOffset p initial c i) (backwardOutputCell P p st.dv i) else t) t
  let t := (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    if (backwardPointer p initial p.dk c).inBounds (TileShape.indexToList _ i) [0,1] then
      t.write p.dk (backwardOffset p initial c i) (backwardOutputCell P p st.dk i) else t) t
  let t := t.setReg "dv_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.dv (c+1))
  t.setReg "dk_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p initial p.dk (c+1))

theorem backwardFinishExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (st : BackwardInner α p)
    (h : BackwardOuterInvariant p initial c st.mem current)
    (hdv : current.regs .real [p.blockM,p.head] "dv"=some st.dv)
    (hdk : current.regs .real [p.blockM,p.head] "dk"=some st.dk)
    (hlo : current.regs .nat [] "lo"=some (fun _ => c*p.blockM))
    (hq : current.regs .blockPtr [p.blockM,p.head] "q_tile_ptr"=some (fun _ => backwardPointer p initial p.q p.blocks))
    (hdo : current.regs .blockPtr [p.blockM,p.head] "do_tile_ptr"=some (fun _ => backwardPointer p initial p.gradient p.blocks))
    (hdq : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _ => backwardPointer p initial p.dq p.blocks)) :
    run (backend rest P) ((backwardOuterBody p).drop 13) current=some (backwardFinishResult P p initial current c st) := by
  simp [backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,evalDeltas,numeric,store,backend,backwardFinishResult,backwardOutputCell,
    h.keyPtr,h.valuePtr,h.dvPtr,h.dkPtr,hdv,hdk,hlo,hq,hdo,hdq,backwardPointer_reset,reset_offset,backwardPointer,backwardOffset,Nat.add_mul,Nat.add_assoc,
    TileShape.indexToList,Region.cast,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  try simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Nat.add_assoc]
  simp only [reset_offset,Nat.add_mul,Nat.one_mul,Nat.add_assoc]
  all_goals rfl

theorem backwardFinishInvariant (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (mem : Memory α)
    (h : BackwardOuterInvariant p initial c
      (backwardInnerRounds P p initial mem c (p.blocks-c)).mem current) :
    BackwardEntryInvariant p initial (c+1) (backwardOuterAdvance P p initial c mem)
      (backwardFinishResult P p initial current c (backwardInnerRounds P p initial mem c (p.blocks-c))) := by
  constructor
  · constructor
    · constructor <;> simp [backwardFinishResult,h.undef,h.pids,h.hz,h.z,h.h,h.strideZ,h.strideH,h.dqBase]
    · simp [backwardFinishResult,maskedScatterMemoryReference,h.memory,backwardOuterAdvance,backwardOutputCell]
    all_goals simp [backwardFinishResult]
  all_goals simp [backwardFinishResult]

#axiomsClean backwardPointer_reset
#axiomsClean backwardFinishExecution
#axiomsClean backwardFinishInvariant
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
