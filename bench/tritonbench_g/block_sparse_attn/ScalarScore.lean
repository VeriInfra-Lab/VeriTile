import bench.tritonbench_g.block_sparse_attn.Logical
import bench.tritonbench_g.block_sparse_attn.Components
import bench.tritonbench_g.block_sparse_attn.ChannelLoop

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

theorem scoreStep_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (column : Int) (nonnegative : 0≤column)
    (old : Matrix α) (block : Nat) (i : TileIndex [16,16]) :
    scoreStep (Arithmetic.ofLegacy a literal argument) Bool.true p s column old block i=
      broadcast (BlockSparseAttn.dot a .mma16n8k8 16
        (fun d => qLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
          (row s i.1) (block*16+d))
        (fun d => kLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
          (column*16+i.2.1.val) (block*16+d)) (old i (owner i.2.1.val))) := by
  have hq : (fun t : Fin 16 => query (Arithmetic.ofLegacy a literal argument) p s block (i.1,t,PUnit.unit))=
      (fun t : Fin 16 => broadcast (qLoad (layoutOf p batches rows cols) a m
        (batch p s) (head p s) (row s i.1) (block*16+t.val))) :=
    funext (fun t => query_view a literal argument zero p batches s m rows cols views block (i.1,t,PUnit.unit))
  have hk : (fun t : Fin 16 => key (Arithmetic.ofLegacy a literal argument) p s column block (t,i.2.1,PUnit.unit))=
      (fun t : Fin 16 => broadcast (kLoad (layoutOf p batches rows cols) a m
        (batch p s) (head p s) (column*16+i.2.1.val) (block*16+t.val))) :=
    funext (fun t => key_view a literal argument zero p batches s m rows cols views column nonnegative block (t,i.2.1,PUnit.unit))
  simp only [scoreStep,ite_true,tensor]
  rw [hq,hk]
  exact congrArg broadcast (chain_broadcast a literal argument
    (fun d => qLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
      (row s i.1) (block*16+d))
    (fun d => kLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
      (column*16+i.2.1.val) (block*16+d)) _)

theorem scoreAccumulation_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (column : Int) (nonnegative : 0≤column)
    (i : TileIndex [16,16]) :
    scoreAccumulation (Arithmetic.ofLegacy a literal argument) Bool.true p s column p.blocks i=
      broadcast (scoreDot (layoutOf p batches rows cols) a m (batch p s) (head p s)
        (row s i.1) (column*16+i.2.1.val)) := by
  have aux (blocks : Nat) :
      scoreAccumulation (Arithmetic.ofLegacy a literal argument) Bool.true p s column blocks i=
      broadcast ((List.range blocks).foldl (fun old block => BlockSparseAttn.dot a .mma16n8k8 16
        (fun d => qLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
          (row s i.1) (block*16+d))
        (fun d => kLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
          (column*16+i.2.1.val) (block*16+d)) old) a.zero) := by
    induction blocks with
    | zero => simp [scoreAccumulation,Arithmetic.ofLegacy,zero]
    | succ blocks ih =>
      rw [scoreAccumulation_succ,scoreStep_view a literal argument zero p batches s m rows cols views column nonnegative]
      rw [ih]
      simp only [broadcast,List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil]
  exact aux p.blocks

theorem scores_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (first0 : 0≤first p s) (visit : Nat)
    (column0 : 0≤csrColumn p s (first p s+visit)) (i : TileIndex [16,16]) :
    Component.scores (Arithmetic.ofLegacy a literal argument) Bool.true p s
        (csrColumn p s (first p s+visit)) i=
      broadcast (score (layoutOf p batches rows cols) a m (argument p.scale)
        (batch p s) (head p s) (row s i.1) visit i.2.1) := by
  funext o
  simp only [Component.scores,ite_true]
  rw [scoreAccumulation_view a literal argument zero p batches s m rows cols views _ column0 i]
  simp only [broadcast,Arithmetic.ofLegacy,zero]
  rw [column_view p batches s m rows cols views first0 i.1 visit]
  rfl

#axiomsClean scoreStep_view
#axiomsClean scoreAccumulation_view
#axiomsClean scores_view
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
