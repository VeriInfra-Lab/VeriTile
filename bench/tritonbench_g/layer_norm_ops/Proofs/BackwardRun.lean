import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude00
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude01
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude10
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude11
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody000
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody001
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody010
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody011
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody100
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody101
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody110
import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardBody111
namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
noncomputable section

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s : State α) :
    ∃ t, run M (prelude p) s = some t ∧ Ready p s t (start p s) (initial M p s) := by
  cases hd : p.hasDres <;> cases hs : p.storeDres
  · exact prelude00 M p s hd hs
  · exact prelude01 M p s hd hs
  · exact prelude10 M p s hd hs
  · exact prelude11 M p s hd hs

theorem body_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready p s t row a) :
    ∃ u, run M (body p) (t.setReg "row" .nat [] (fun _ => row)) = some u ∧
      Ready p s u (row+1) (update M p row a) := by
  cases hr : p.rms <;> cases hd : p.hasDres <;> cases hs : p.storeDres
  · exact body000 M p s t row a h hr hd hs
  · exact body001 M p s t row a h hr hd hs
  · exact body010 M p s t row a h hr hd hs
  · exact body011 M p s t row a h hr hd hs
  · exact body100 M p s t row a h hr hd hs
  · exact body101 M p s t row a h hr hd hs
  · exact body110 M p s t row a h hr hd hs
  · exact body111 M p s t row a h hr hd hs

#axiomsClean prelude_run
#axiomsClean body_run
end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
