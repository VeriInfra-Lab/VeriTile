import bench.tritonbench_g.mixed_sparse_attention.IO
import bench.tritonbench_g.mixed_sparse_attention.Proofs.PublicReal

namespace VeriTile.Bench.Repairs.MixedSparseAttention.LogicalIOProof
open VeriTile Triton FP.Structural
open Whole002 Source Whole
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

theorem headline_iff (c : Config) (work : List Program)
    (formula : KernelIO.LogicalInputs → Fin (c.Z*c.H) → Fin c.T → Fin c.BD → ℝ)
    (same : ∀ xs hz row col, formula (KernelIO.namedInputs c xs) hz row col =
      RealIO.mathematical c xs (0 : Fin 1) (hz,row,col,PUnit.unit)) :
    (RealIO.logicalIO c work ⊨{RealIO.logicalDomain c} formula) ↔
    (RealIO.io c work).ImplementsOn (RealIO.domain c) (RealIO.mathematical c) := by
  have domain_eq : (fun xs => RealIO.logicalDomain c (KernelIO.namedInputs c xs)) = RealIO.domain c := rfl
  have target : (RealIO.logicalIO c work).lift formula = RealIO.mathematical c := by
    funext xs port i
    fin_cases port
    exact same xs i.1 i.2.1 i.2.2.1
  change (RealIO.io c work).ImplementsOn
    (fun xs => RealIO.logicalDomain c (KernelIO.namedInputs c xs))
    ((RealIO.logicalIO c work).lift formula) ↔ _
  exact (congrArg₂ (fun domain target => (RealIO.io c work).ImplementsOn domain target) domain_eq target).to_iff

theorem implements (c : Config) (sep : Separated c) (layout : Contiguous c)
    (hbm : 0<c.BM) (hbn : 0<c.BN) (work : List Program)
    (order : work.Perm (programs (grid c)))
    (formula : KernelIO.LogicalInputs → Fin (c.Z*c.H) → Fin c.T → Fin c.BD → ℝ)
    (same : ∀ xs hz row col, formula (KernelIO.namedInputs c xs) hz row col =
      RealIO.mathematical c xs (0 : Fin 1) (hz,row,col,PUnit.unit)) :
    (RealIO.logicalIO c work ⊨{RealIO.logicalDomain c} formula) :=
  (headline_iff c work formula same).2 (RealIO.implements c sep layout hbm hbn work order)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MixedSparseAttention.LogicalIOProof
