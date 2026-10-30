import bench.tritonbench_g.layer_norm_fwd.IO
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealImplementation

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open scoped VeriTile.Triton.LogicalIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000

@[simp] theorem flat_div {M N : Nat} (row : Fin M) (col : Fin N) :
    (finProdFinEquiv (row,col) : Fin (M*N)).val / N = row.val := by
  exact congrArg (fun q : Fin M × Fin N => q.1.val)
    ((finProdFinEquiv : Fin M × Fin N ≃ Fin (M*N)).symm_apply_apply (row,col))

@[simp] theorem flat_mod {M N : Nat} (row : Fin M) (col : Fin N) :
    (finProdFinEquiv (row,col) : Fin (M*N)).val % N = col.val := by
  exact congrArg (fun q : Fin M × Fin N => q.2.val)
    ((finProdFinEquiv : Fin M × Fin N ≃ Fin (M*N)).symm_apply_apply (row,col))

def mathematical (f : Flags) (M N : Nat) (eps p : ℝ) (a : Inputs M N) : Outputs M N :=
  let x := mathInputs M N a
  ⟨fun row col => Formula.y f x M N eps p row.val col.val,
   fun row col => Formula.y1 f x M N eps p row.val col.val,
   fun row => Formula.mean f x M N p row.val,
   fun row => Formula.rstd f x M N eps p row.val,
   fun row col => Formula.combined f x M p row.val col.val,
   fun row col => Formula.keep (x.seeds row.val) p col.val,
   fun row col => Formula.keep (x.seeds (M+row.val)) p col.val⟩

theorem lift_formula (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) :
    (io f r stride M N eps p).lift (mathematical f M N eps p) =
      RealInterface.formula f M N eps p := by
  funext a
  change (outputs M N).symm (mathematical f M N eps p ((inputs M N) a)) = _
  apply (outputs M N).injective
  rw [Equiv.apply_symm_apply]
  simp only [mathematical,mathInputs,Equiv.symm_apply_apply]
  simp only [outputs,Equiv.coe_fn_mk,RealInterface.formula,flat_div,flat_mod]
  rfl

theorem judgment_iff (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) :
    (io f r stride M N eps p ⊨ mathematical f M N eps p) ↔
      (RealInterface.io f r stride M N eps p).Implements
        (RealInterface.formula f M N eps p) := by
  change (RealInterface.io f r stride M N eps p).Implements _ ↔ _
  rw [lift_formula]

theorem implements (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (width : widthCheck N=Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N) :
    io f r stride M N eps p ⊨ mathematical f M N eps p :=
  (judgment_iff f r stride M N eps p).mpr
    (RealInterface.implements f r stride M N eps p width sep geometry)
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalIO
