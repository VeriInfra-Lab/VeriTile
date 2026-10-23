import bench.repairs.tritonbench_g.mixed_sparse_attention.NativeIO
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Native
noncomputable section
set_option maxHeartbeats 2000000

theorem query_source (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c)
    (s : State Expr) (h : KernelIO.Reads c xs s) (p : Program) :
    Source.queryValues (algebra rt) c (start (grid c) p s)=query rt c xs p := by
  unfold Source.queryValues query
  rw [show (algebra rt).fp32Load=id from rfl]
  simp only [Source.rowStart,Source.sequenceLength,Source.queryBase,start,ids,
    if_true,if_false,show ¬(1:Nat)=0 by decide,h.query,h.length,
    KernelIO.row,KernelIO.sequence,KernelIO.qbase,KernelIO.length]
  rfl

theorem key_source (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c)
    (s : State Expr) (h : KernelIO.Reads c xs s) (p : Program) (ph : Bool) (n : Nat) :
    Source.keyValues (algebra rt) c (start (grid c) p s) ph n=key rt c xs p ph n := by
  funext i
  unfold Source.keyValues key
  by_cases active : KernelIO.keyActive c xs p ph n i.2.1
  · have original := (KernelIO.Reads.keyActive c xs s h p ph n i.2.1).mpr active
    simp only [if_pos active,if_pos original]
    rw [KernelIO.Reads.selected c xs s h p ph n i.2.1]
    simp only [Source.keyValueBase,start,ids,if_true,if_false,show ¬(1:Nat)=0 by decide,
      h.key,KernelIO.kvbase]
  · have original : ¬Source.keyActive c (start (grid c) p s) ph n i.2.1 :=
      fun hraw=>active ((KernelIO.Reads.keyActive c xs s h p ph n i.2.1).mp hraw)
    simp only [if_neg active,if_neg original]

theorem value_source (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c)
    (s : State Expr) (h : KernelIO.Reads c xs s) (p : Program) (ph : Bool) (n : Nat) :
    Source.valueValues (algebra rt) c (start (grid c) p s) ph n=value rt c xs p ph n := by
  funext i
  unfold Source.valueValues value
  by_cases active : KernelIO.keyActive c xs p ph n i.1
  · have original := (KernelIO.Reads.keyActive c xs s h p ph n i.1).mpr active
    simp only [if_pos active,if_pos original]
    rw [KernelIO.Reads.selected c xs s h p ph n i.1]
    simp only [Source.keyValueBase,start,ids,if_true,if_false,show ¬(1:Nat)=0 by decide,
      h.value,KernelIO.kvbase]
  · have original : ¬Source.keyActive c (start (grid c) p s) ph n i.1 :=
      fun hraw=>active ((KernelIO.Reads.keyActive c xs s h p ph n i.1).mp hraw)
    simp only [if_neg active,if_neg original]

theorem step_source (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (xs : KernelIO.Values Expr c) (s : State Expr) (h : KernelIO.Reads c xs s)
    (p : Program) (ph : Bool) (n : Nat) :
    Source.update (algebra rt) (dotModel c.storage I) c (start (grid c) p s) ph n=
      step rt I c xs p ph n := by
  classical
  funext a
  have mask (i : TileIndex [c.BM,c.BN]) :
      (Source.rowStart c (start (grid c) p s)+i.1.val<Source.sequenceLength c (start (grid c) p s) ∧
        (if ph then Source.keyActive c (start (grid c) p s) Bool.true n i.2.1
         else Source.keyActive c (start (grid c) p s) Bool.false n i.2.1 ∧
           Source.columnValues c (start (grid c) p s) Bool.false n i.2.1≤Source.rowStart c (start (grid c) p s)+i.1.val)) ↔
      (KernelIO.row c p+i.1.val<KernelIO.sequence c xs p ∧
        (if ph then KernelIO.keyActive c xs p Bool.true n i.2.1
         else KernelIO.keyActive c xs p Bool.false n i.2.1 ∧
           KernelIO.selected c xs p Bool.false n i.2.1≤KernelIO.row c p+i.1.val)) := by
    have row : Source.rowStart c (start (grid c) p s)=KernelIO.row c p := rfl
    have length : Source.sequenceLength c (start (grid c) p s)=KernelIO.sequence c xs p :=
      h.length _
    rw [row,length]
    cases ph <;> simp only [Bool.true_eq_false,if_true,if_false,
      KernelIO.Reads.keyActive c xs s h p,KernelIO.Reads.selected c xs s h p]
  have scoreMask (i : TileIndex [c.BM,c.BN]) :
      (if Source.rowStart c (start (grid c) p s)+i.1.val<Source.sequenceLength c (start (grid c) p s) ∧
        (if ph then Source.keyActive c (start (grid c) p s) Bool.true n i.2.1
         else Source.keyActive c (start (grid c) p s) Bool.false n i.2.1 ∧
           Source.columnValues c (start (grid c) p s) Bool.false n i.2.1≤Source.rowStart c (start (grid c) p s)+i.1.val)
       then (algebra rt).literal (some .fp32) .real 0 else (algebra rt).negInf)=
      (if KernelIO.row c p+i.1.val<KernelIO.sequence c xs p ∧
        (if ph then KernelIO.keyActive c xs p Bool.true n i.2.1
         else KernelIO.keyActive c xs p Bool.false n i.2.1 ∧
           KernelIO.selected c xs p Bool.false n i.2.1≤KernelIO.row c p+i.1.val)
       then (algebra rt).literal (some .fp32) .real 0 else (algebra rt).negInf) := by
    by_cases active : KernelIO.row c p+i.1.val<KernelIO.sequence c xs p ∧
      (if ph then KernelIO.keyActive c xs p Bool.true n i.2.1
       else KernelIO.keyActive c xs p Bool.false n i.2.1 ∧
         KernelIO.selected c xs p Bool.false n i.2.1≤KernelIO.row c p+i.1.val)
    · have original := (mask i).mpr active
      simp only [if_pos original,if_pos active]
    · have original := mt (mask i).mp active
      simp only [if_neg original,if_neg active]
  simp only [Source.update,step,query_source rt c xs s h p,key_source rt c xs s h p,
    value_source rt c xs s h p,scoreMask]

theorem accumulate_source (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (xs : KernelIO.Values Expr c) (s : State Expr) (h : KernelIO.Reads c xs s) (p : Program) :
    Source.finalValues (algebra rt) (dotModel c.storage I) c (start (grid c) p s)=
      accumulate rt I c xs p := by
  have eq : Source.update (algebra rt) (dotModel c.storage I) c (start (grid c) p s)=
      step rt I c xs p := by
    funext ph n
    exact step_source rt I c xs s h p ph n
  unfold Source.finalValues accumulate
  rw [eq]
  simp only [Source.blockCount,Source.columnCount,
    Source.metadataIndex,start,ids,if_true,if_false,show ¬(1:Nat)=0 by decide,
    h.blocks,h.counts,KernelIO.blockCount,KernelIO.columnCount,KernelIO.slot]
  rfl

theorem expression_source (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (xs : KernelIO.Values Expr c) (s : State Expr) (h : KernelIO.Reads c xs s) (p : Program)
    (i : TileIndex [c.BM,c.BD]) :
    Whole.answer (algebra rt) (dotModel c.storage I) c (start (grid c) p s) i=
      expression rt I c xs p i := by
  simp only [Whole.answer,accumulate_source rt I c xs s h p,Source.outputValue,expression]

#axiomsClean expression_source
end
end VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
