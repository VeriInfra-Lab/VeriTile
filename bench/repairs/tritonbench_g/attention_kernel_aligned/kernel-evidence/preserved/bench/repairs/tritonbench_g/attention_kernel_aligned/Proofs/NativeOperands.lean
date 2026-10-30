import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.NativeProjection
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Native
open VeriTile Triton FP.Structural Source Whole Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Little-endian half-word read from the current finite caller allocation.
The proof argument derives byte bounds from TensorStorage and Placement. -/
def readHalf (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (flat : Fin pl.arena → BitVec 8) (r : Buffer) (off : Nat)
    (ho : off<extent storage r) : Word :=
  let lo : Fin pl.arena := ⟨pl.base r+2*off,by have:=pl.bounded r;omega⟩
  let hi : Fin pl.arena := ⟨pl.base r+2*off+1,by have:=pl.bounded r;omega⟩
  BitVec.ofNat 32 ((flat lo).toNat+256*(flat hi).toNat)

theorem half_word_decomposition (w : BitVec 16) :
    BitVec.ofNat 32
      ((BitVec.ofNat 8 ((w.zeroExtend 32).toNat)).toNat+
       256*(BitVec.ofNat 8 ((w.zeroExtend 32).toNat/256)).toNat)=w.zeroExtend 32 := by
  apply BitVec.eq_of_toNat_eq
  have hw:=w.isLt
  simp only [BitVec.toNat_ofNat,BitVec.zeroExtend_eq_setWidth,BitVec.toNat_setWidth]
  norm_num at hw ⊢
  omega

/-- No precomputed operand is admitted: every live raw load is reconstructed
from the two currently bound bytes, including storage-offset pointers. -/
theorem input_from_current_bytes (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (s : State Word)
    (binding : WordBinding p storage pl flat s) (inputs : RawInputs p s)
    (r : Buffer) (hr : r≠.output) (off : Nat) (ho : off<extent storage r) :
    s.mem (region p r) off=.mk .fp16 (readHalf p storage pl flat r off ho) := by
  have hin : region p r∈[p.q,p.k,p.v,p.bias] := by cases r <;> simp_all [region]
  obtain ⟨w,hw⟩:=inputs _ hin off
  let lo : Fin pl.arena := ⟨pl.base r+2*off,by have:=pl.bounded r;omega⟩
  let hi : Fin pl.arena := ⟨pl.base r+2*off+1,by have:=pl.bounded r;omega⟩
  have hl:=binding.inputs r hr off ho 0 (by decide) lo (by simp [lo])
  have hh:=binding.inputs r hr off ho 1 (by decide) hi rfl
  rw [hw] at hl hh ⊢
  simp only [rawCell,pow_zero,pow_one,Nat.div_one] at hl hh
  change Cell.mk .fp16 (w.zeroExtend 32)=Cell.mk .fp16
    (BitVec.ofNat 32 ((flat lo).toNat+256*(flat hi).toNat))
  rw [hl,hh,half_word_decomposition]

/-- Total notation for a current-memory operand. Active source accesses are
proved in range below; the out-of-range arm is never used by a live load. -/
def currentExpr (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (flat : Fin pl.arena → BitVec 8) (r : Buffer) (off : Nat) : Expr :=
  if ho : off<extent storage r then fun _=>.ready (.bits (readHalf p storage pl flat r off ho))
  else fun _=>.zero

theorem current_read (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (s : State Word)
    (binding : WordBinding p storage pl flat s) (inputs : RawInputs p s)
    (r : Buffer) (hr : r≠.output) (off : Nat) (ho : off<extent storage r) :
    ((injectState s).mem (region p r) off).read .fp16=currentExpr p storage pl flat r off := by
  have hw:=input_from_current_bytes p storage pl flat s binding inputs r hr off ho
  change (match s.mem (region p r) off with | .mk d v=>Cell.mk d (inject d v)).read .fp16=_
  rw [hw]
  simp only [inject,Cell.read_same,currentExpr,dif_pos ho]

/-- The five source loads, including inactive tails, are bound to the current
bytes. This is an operand statement, with no final value or execution premise. -/
def CurrentOperands (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (flat : Fin pl.arena → BitVec 8) (s : State Word) : Prop :=
  (∀ i : QLane p,qValue (modelFor p) p (injectState s) i=
    if row p (injectState s) i.1<p.length then
      currentExpr p storage pl flat .query (qOffset p (injectState s) i) else fun _=>.zero) ∧
  (∀ c (i : KLane p),kValue (modelFor p) p (injectState s) c i=
    if c*p.blockN+i.2.1.val<p.length then
      currentExpr p storage pl flat .key (kOffset p (injectState s) c i) else fun _=>.zero) ∧
  (∀ c (i : VLane p),vValue (modelFor p) p (injectState s) c i=
    if c*p.blockN+i.1.val<p.length then
      currentExpr p storage pl flat .value (vOffset p (injectState s) c i) else fun _=>.zero) ∧
  (∀ c,c<p.blocks → ∀ i : Fin p.blockM,b0Value (modelFor p) p (injectState s) c i=
    if row p (injectState s) i<p.length then
      currentExpr p storage pl flat .bias (b0Offset p (injectState s) c i) else fun _=>.zero) ∧
  (∀ i : ScoreLane p,b1Value (modelFor p) p (injectState s) i=
    if row p (injectState s) i.1<p.length then
      currentExpr p storage pl flat .bias (b1Offset p (injectState s) i) else fun _=>.zero)

theorem source_current_operands (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (s : State Word)
    (binding : WordBinding p storage pl flat s) (inputs : RawInputs p s)
    (hy : s.pids 1<p.batches*p.heads) : CurrentOperands p storage pl flat s := by
  obtain ⟨_,hb,hh⟩:=head_decode p (s.pids 1) hy
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro i
    unfold qValue
    split_ifs with hi
    · apply current_read p storage pl flat s binding inputs .query (by decide)
      simpa [qOffset,headBase,offset,qStrides] using
        storage.query _ hb _ hh _ hi _ i.2.1.isLt
    · simp [modelFor,model,algebra,literal,narrow]
  · intro c i
    unfold kValue
    split_ifs with hi
    · apply current_read p storage pl flat s binding inputs .key (by decide)
      simpa [kOffset,kvBase,offset,kStrides,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using
        storage.key _ hb _ hh _ hi _ i.1.isLt
    · simp [modelFor,model,algebra,literal,narrow]
  · intro c i
    unfold vValue
    split_ifs with hi
    · apply current_read p storage pl flat s binding inputs .value (by decide)
      simpa [vOffset,valueBase,offset,vStrides] using
        storage.value _ hb _ hh _ hi _ i.2.1.isLt
    · simp [modelFor,model,algebra,literal,narrow]
  · intro c hc i
    unfold b0Value
    split_ifs with hi
    · apply current_read p storage pl flat s binding inputs .bias (by decide)
      have hcap : c<128 := by
        have hn:=blocks_capacity p guard
        omega
      simpa [b0Offset,biasBase,offset,bStrides] using storage.bias _ hb _ hh _ hi _ hcap
    · simp [modelFor,model,algebra,literal,narrow]
  · intro i
    unfold b1Value
    split_ifs with hi
    · apply current_read p storage pl flat s binding inputs .bias (by decide)
      have hlast : i.2.1.val%p.biasLast+p.biasLast<128 := by
        rw [guard.selected.biasLast]
        omega
      simpa [b1Offset,biasBase,offset,bStrides] using storage.bias _ hb _ hh _ hi _ hlast
    · simp [modelFor,model,algebra,literal,narrow]

theorem grid_current_operands (p : Parameters) (guard : Guard p) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (s : State Word)
    (binding : WordBinding p storage pl flat s) (inputs : RawInputs p s)
    (job : Program) (hjob : job∈programs (grid p)) :
    CurrentOperands p storage pl flat (start (grid p) job s) := by
  have hy:=(mem_programs (grid p) job.1 job.2.1 job.2.2).mp hjob
  apply source_current_operands p guard storage pl flat (start (grid p) job s)
    ⟨binding.inputs,⟨binding.output.output⟩⟩ inputs
  simpa [start,ids,grid] using hy.2.1

#axiomsClean input_from_current_bytes
#axiomsClean source_current_operands
#axiomsClean grid_current_operands
#specNonCircular CurrentOperands avoiding [invoke,runPrograms,FP.Structural.exec,outputGraph]
#specNonCircular readHalf avoiding [invoke,runPrograms,FP.Structural.exec,outputGraph]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Native
