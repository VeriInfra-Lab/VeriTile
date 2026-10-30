import bench.repairs.tritonbench_g.mul_exponent_compensator.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Commuted
open VeriTile.Triton FP.Structural Interface
noncomputable section

def invocation {α : Type} (M : Algebra α) (src dst : RegionName)
    (n B pid : Nat) : Invocation α :=
  { Interface.invocation M src dst n B pid with kernel := scalarLeftKernel src dst n B }

def variant {α : Type} (M : Algebra α) (src dst : RegionName) (n B : Nat)
    (order : List Nat) : ProgramIO.Variant (io M src dst n B order) where
  schedule := fun _ => order.map (invocation M src dst n B)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Commuted
