import bench.tritonbench_g.block_sparse_attn.ScalarComplete
import bench.tritonbench_g.block_sparse_attn.RealUpdate
import bench.tritonbench_g.block_sparse_attn.Proofs.RealComplete

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

theorem output_real (p : Parameters) (batches : Nat) (s : State (Replicas R))
    (m : BlockSparseAttn.Memory ℝ) (rows cols : Int → Int)
    (views : Views s (BlockSparseAttn.Real.lift m) rows cols) (first0 : 0≤first p s)
    (columns : ∀ visit,visit<visitCount p s → 0≤csrColumn p s (first p s+visit))
    (i : TileIndex [16,16]) (live : row s i.1<p.length) :
    output mathArithmetic Bool.false p s i (owner i.2.1.val)=
      some (Spec.output (layoutOf p batches rows cols).shape (layoutOf p batches rows cols).csr
        (BlockSparseAttn.Real.inputs (layoutOf p batches rows cols) m) p.scale
        (batch p s) (head p s) (row s i.1) (channel s i.2.1)) := by
  rw [output_math]
  have negativeInfinity : BlockSparseAttn.Real.ops.add BlockSparseAttn.Real.ops.zero
      BlockSparseAttn.Real.ops.negInf=BlockSparseAttn.Real.ops.negInf :=
    rfl
  unfold mathArithmetic
  rw [output_view BlockSparseAttn.Real.ops (fun x => some x) (fun x => some x) rfl negativeInfinity
    p batches s (BlockSparseAttn.Real.lift m) rows cols views first0 columns i]
  exact BlockSparseAttn.Real.row_real (layoutOf p batches rows cols) m p.scale (work p s i) live (by norm_num [layoutOf]) (by norm_num [layoutOf])

#axiomsClean output_real
#specNonCircular Spec.output avoiding [output,accumulation,update]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
