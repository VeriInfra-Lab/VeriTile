import bench.repairs.tritonbench_g.adam_update_triton.Proofs.IOReal
import bench.repairs.tritonbench_g.adam_update_triton.IO

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.LogicalReader
open VeriTile.Triton FP.Structural
noncomputable section

def canonical (params : Scalars ℝ) (B : Nat) (values : Interface.Inputs ℝ B) :=
  Interface.result
    (fun j => Math.target params .parameter (Interface.parameter values j)
      (Interface.gradient values j) (Interface.momentum values j))
    (fun j => Math.target params .momentum (Interface.parameter values j)
      (Interface.gradient values j) (Interface.momentum values j))

theorem statement_iff (params : Scalars ℝ) (total n B pid : Nat)
    (f : LogicalIO.Inputs B → LogicalIO.Outputs B)
    (connection : (LogicalIO.io params total n B pid).lift f = canonical params B) :
    (LogicalIO.io params total n B pid).Implements f ↔
      (Interface.programIO Real.semantics params id id total n B pid).Implements (canonical params B) := by
  change (Interface.programIO Real.semantics params id id total n B pid).Implements
    ((LogicalIO.io params total n B pid).lift f) ↔ _
  rw [connection]

theorem correct (params : Scalars ℝ) (total n B pid : Nat)
    (f : LogicalIO.Inputs B → LogicalIO.Outputs B)
    (connection : (LogicalIO.io params total n B pid).lift f = canonical params B) :
    (LogicalIO.io params total n B pid).Implements f :=
  (statement_iff params total n B pid f connection).mpr (Real.io_correct params total n B pid)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.LogicalReader
