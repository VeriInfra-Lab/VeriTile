import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Execution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Only the two endpoint cells actually loaded by this program. -/
def RowInt32 {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  Int32Bits.inRange (first p s) ∧ Int32Bits.inRange (last p s)

/-- Only the current CSR occurrence is used by the signed cast. -/
def ColumnInt32 {α : Type} [Inhabited α] (p : Parameters) (s : State α) (slot : Int) : Prop :=
  Int32Bits.inRange (csrColumn p s slot)

/-- Finite actual metadata reads. Unread cells need no signed-range premise. -/
structure ReadInt32 {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop where
  rows : RowInt32 p s
  columns : ∀ visit,visit<visitCount p s → ColumnInt32 p s (first p s+visit)

def prefixExecution_local {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (integers : RowInt32 p s) :
    {t : State α // run M (prelude p) s = some t} := by
  have rows0 := integers.1
  have rows1 := integers.2
  dsimp only [first,last,rowPointer,layout,head] at rows0 rows1
  refine ⟨?_, ?_⟩
  swap
  simp [prelude,source,declared,ComputeKernel.surfaceBody,run,FP.Structural.step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Int32Bits.valuesValid,rows0,rows1]
  rfl

theorem prefixIndices_local {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State (Replicas α)) (integers : RowInt32 p s) :
    Indices a p s (prefixExecution_local a.model p (bindScale (broadcast (a.argument p.scale)) s) integers).val := by
  constructor <;>
    simp [prefixExecution_local,bindScale,batch,head,kv,layout,row,channel,qPointer,kPointer,vPointer,
      first,last,rowPointer,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem initialRuntime_local {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State (Replicas α)) (integers : RowInt32 p s) :
    Runtime a p s (initialCarry a) (prefixExecution_local a.model p (bindScale (broadcast (a.argument p.scale)) s) integers).val := by
  refine ⟨prefixIndices_local a p s integers,?_,?_,?_⟩ <;>
    simp [prefixExecution_local,bindScale,initialCarry,Arithmetic.model,binary,broadcast]
  funext i
  rfl

def visitPrefixExecution_local {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : ColumnInt32 p initial slot) :
    {t : State (Replicas α) // run a.model ((body p).take 3)
      (current.setReg "col_idx_idx" .int [] (fun _ => slot)) = some t} := by
  have cols := integers
  dsimp only [ColumnInt32,csrColumn] at cols
  refine ⟨?_,?_⟩
  swap
  simp [body,source,declared,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Arithmetic.model,Int32Bits.valuesValid,
    h.scale,h.memory,h.layout,cols,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem visitPrefixInvariant_local {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : ColumnInt32 p initial slot) :
    ChannelInvariant a p initial carry (csrColumn p initial slot)
      (fun _ => broadcast (a.literal 0))
      (visitPrefixExecution_local a p initial current carry slot h integers).val := by
  refine ⟨⟨⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩,?_,?_,?_⟩,?_,?_⟩
  all_goals simp [visitPrefixExecution_local,csrColumn,Arithmetic.model,
    h.scale,h.memory,h.length,h.batch,h.head,h.channel,h.rows,h.keys,h.dimensions,
    h.qptrs,h.kptrs,h.vptrs,h.layout,h.first,h.last,h.maximum,h.denominator,h.accumulator,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

theorem visitExecution_local {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : ColumnInt32 p initial slot) :
    ∃ final, run a.model (loopBody native p)
      (current.setReg "col_idx_idx" .int [] (fun _ => slot))=some final ∧
      Runtime a p initial (update a native p initial (csrColumn p initial slot) carry) final := by
  let start := visitPrefixExecution_local a p initial current carry slot h integers
  have hi : ChannelInvariant a p initial carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) 0) start.val :=
    visitPrefixInvariant_local a p initial current carry slot h integers
  obtain ⟨mid,hm,hmi⟩ := channelLoopExecution a native p initial start.val carry
    (csrColumn p initial slot) 0 (by omega) hi
  let finish := visitSuffixExecution a native p initial mid carry (csrColumn p initial slot) hmi
  have hloop : run a.model [.forLoop "score_block" p.blocks (channelBody native p)] start.val=some mid := by
    simp [run,FP.Structural.step,hm]
  refine ⟨finish.val,?_,visitSuffixInvariant a native p initial mid carry (csrColumn p initial slot) hmi⟩
  rw [loopBody_partition,run_append,run_append,start.property]
  simp only [Option.bind_some,hloop,finish.property]

theorem outerLoopExecution_local {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (visit : Nat) (bound : visit≤visitCount p initial)
    (h : Runtime a p initial (accumulation a native p initial visit) current)
    (integers : ∀ visit,visit<visitCount p initial → ColumnInt32 p initial (first p initial+visit)) :
    ∃ final, rangeInt a.model "col_idx_idx" (first p initial+visit) (last p initial) 1
      (loopBody native p) current=some final ∧
      Runtime a p initial (accumulation a native p initial (visitCount p initial)) final := by
  have aux : ∀ n k (current : State (Replicas α)), visitCount p initial-k=n →
      k≤visitCount p initial → Runtime a p initial (accumulation a native p initial k) current →
      ∃ final, rangeInt a.model "col_idx_idx" (first p initial+k) (last p initial) 1
        (loopBody native p) current=some final ∧
        Runtime a p initial (accumulation a native p initial (visitCount p initial)) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases less : k<visitCount p initial
      · obtain ⟨next,he,hi⟩ := visitExecution_local a native p initial current
          (accumulation a native p initial k) (first p initial+k) h (integers k less)
        rw [← accumulation_succ] at hi
        obtain ⟨final,hf,hfi⟩ := ih (visitCount p initial-(k+1)) (by omega)
          (k+1) next rfl (by omega) hi
        refine ⟨final,?_,hfi⟩
        have active : first p initial+(k : Int)<last p initial := by
          have positive : 0<last p initial-first p initial := by
            by_contra nonpositive
            have empty : visitCount p initial=0 :=
              Int.toNat_eq_zero.mpr (le_of_not_gt nonpositive)
            omega
          have cast : ((visitCount p initial:Nat):Int)=last p initial-first p initial :=
            Int.toNat_of_nonneg (le_of_lt positive)
          have lt : (k:Int)<(visitCount p initial:Int) := Int.ofNat_lt.mpr less
          rw [cast] at lt
          simpa only [Int.add_comm] using (lt_sub_iff_add_lt.mp lt)
        rw [rangeInt,if_neg (by decide),if_pos active,he]
        simpa only [Option.bind_some,Int.natCast_add,Int.natCast_one,Int.add_assoc] using hf
      · have equal : k=visitCount p initial := by omega
        subst k
        refine ⟨current,?_,h⟩
        have inactive : ¬first p initial+(visitCount p initial : Int)<last p initial := by
          have cast : last p initial-first p initial≤(visitCount p initial:Int) :=
            Int.self_le_toNat _
          exact not_lt.mpr (by simpa only [Int.add_comm] using (sub_le_iff_le_add.mp cast))
        rw [rangeInt,if_neg (by decide),if_neg inactive]
  exact aux _ _ _ rfl bound h

theorem complete_local {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State (Replicas α)) (integers : ReadInt32 p initial) :
    ∃ final, execution a native p initial=some final ∧
      final.mem=outputMemory p initial (output a native p initial) := by
  let start := prefixExecution_local a.model p (bindScale (broadcast (a.argument p.scale)) initial) integers.rows
  have hi : Runtime a p initial (accumulation a native p initial 0) start.val :=
    initialRuntime_local a p initial integers.rows
  obtain ⟨mid,hm,hmi⟩ := outerLoopExecution_local a native p initial start.val 0 (by omega) hi integers.columns
  let finish := epilogueExecution a p initial mid
    (accumulation a native p initial (visitCount p initial)) hmi
  have loopRun : run a.model
      [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
        (.ref .int [] "end_l") (.constNat 1) (loopBody native p)] start.val=some mid := by
    simpa [run,FP.Structural.step,evalOp_unfold,hi.first,hi.last] using hm
  refine ⟨finish.val,?_,epilogueMemory a p initial mid _ hmi⟩
  unfold execution
  rw [execution_partition,run_append,run_append,start.property]
  simp only [Option.bind_some,loopRun,finish.property]

theorem Int32Memory.toReadInt32 {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (h : Int32Memory s) : ReadInt32 p s := by
  refine ⟨⟨h "rows" (by simp) _,h "rows" (by simp) _⟩,?_⟩
  intro visit _
  exact h "cols" (by simp) _

#axiomsClean complete_local
#axiomsClean Int32Memory.toReadInt32
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
