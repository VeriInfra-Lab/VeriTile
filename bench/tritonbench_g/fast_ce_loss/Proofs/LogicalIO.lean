import bench.tritonbench_g.fast_ce_loss.IO
import bench.tritonbench_g.fast_ce_loss.Proofs.PublicRealForward
import bench.tritonbench_g.fast_ce_loss.Proofs.PublicRealChunked
import bench.tritonbench_g.fast_ce_loss.Proofs.PublicRealBackward

namespace VeriTile.Bench.Repairs.FastCeLoss.LogicalIO
open VeriTile Triton FP.Structural KernelSource
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem forward_iff (p : Parameters) (row chunk : Nat) (chunked : Bool)
    (domain : ForwardInputs p → Prop) (f : ForwardInputs p → ForwardOutputs) :
    (forwardView KernelReal.algebra RealIO.decode p row chunk chunked ⊨{domain} f) ↔
      (PublicIO.forwardIO KernelReal.algebra some RealIO.decode p row chunk chunked Bool.true).ImplementsOn
        (fun raw => domain ⟨PublicIO.forwardLabel p Bool.true raw,fun j => raw (1 : Fin 3) (j,()),raw (2 : Fin 3) ()⟩)
        (fun raw => let ys := f ⟨PublicIO.forwardLabel p Bool.true raw,fun j => raw (1 : Fin 3) (j,()),raw (2 : Fin 3) ()⟩
          Fin.cases (fun _ => ys.lse) (Fin.cases (fun _ => ys.loss) (fun i => i.elim0))) := Iff.rfl

theorem backward_iff (p : Parameters) (row chunk : Nat) (f : BackwardInputs p → Fin p.block → ℝ) :
    (backwardView KernelReal.algebra RealIO.decode p row chunk ⊨ f) ↔
      (PublicIO.backwardIO KernelReal.algebra some RealIO.decode p row chunk Bool.true).Implements
        (fun raw => Fin.cases (fun j => f ⟨PublicIO.backwardLabel p Bool.true raw,
          fun k => raw (1 : Fin 4) (k,()),raw (2 : Fin 4) (),raw (3 : Fin 4) ()⟩ j.1) (fun i => i.elim0)) := Iff.rfl
end
end VeriTile.Bench.Repairs.FastCeLoss.LogicalIO
