import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.ScalarUpdate
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Loop

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

theorem accumulation_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (negativeInfinity : a.add a.zero a.negInf=a.negInf)
    (p : Parameters) (batches : Nat) (s : State (Replicas α))
    (m : BlockSparseAttn.Memory α) (rows cols : Int → Int) (views : Views s m rows cols)
    (first0 : 0≤first p s)
    (columns : ∀ visit,visit<visitCount p s → 0≤csrColumn p s (first p s+visit))
    (visits : Nat) (bound : visits≤visitCount p s) (i : TileIndex [16,16]) :
    rowCarry (accumulation (Arithmetic.ofLegacy a literal argument) Bool.true p s visits) i=
      (List.range visits).foldl (blockStep (layoutOf p batches rows cols) a m (argument p.scale)
        (batch p s) (head p s) (row s i.1) (channel s i.2.1)) (initial a 4) := by
  induction visits with
  | zero =>
    simp [accumulation,initialCarry,rowCarry,initial,broadcast,Arithmetic.ofLegacy,zero,negativeInfinity]
    exact ⟨rfl,rfl⟩
  | succ visits ih =>
    rw [accumulation_succ,
      update_view a literal argument zero p batches s m rows cols views first0 visits (columns visits (by omega))]
    rw [ih (by omega)]
    simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil]

theorem output_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (negativeInfinity : a.add a.zero a.negInf=a.negInf)
    (p : Parameters) (batches : Nat) (s : State (Replicas α))
    (m : BlockSparseAttn.Memory α) (rows cols : Int → Int) (views : Views s m rows cols)
    (first0 : 0≤first p s)
    (columns : ∀ visit,visit<visitCount p s → 0≤csrColumn p s (first p s+visit))
    (i : TileIndex [16,16]) :
    output (Arithmetic.ofLegacy a literal argument) Bool.true p s i (owner i.2.1.val)=
      rowValue (layoutOf p batches rows cols) a m (argument p.scale) (work p s i) := by
  have result := congrArg BlockSparseAttn.Acc.output
    (accumulation_view a literal argument zero negativeInfinity p batches s m rows cols views first0 columns
      (visitCount p s) (Nat.le_refl _) i)
  simpa only [output,rowCarry,rowValue,work,← count_view p batches s m rows cols views i.1] using result

#axiomsClean accumulation_view
#axiomsClean output_view
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
