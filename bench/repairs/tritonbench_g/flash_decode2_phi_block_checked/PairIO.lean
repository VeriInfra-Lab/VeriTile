import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.IO
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.IOPair
open VeriTile Triton FP.Structural FlashDecode2
noncomputable section

def io (g : PhiIO.Geometry) (order : List Nat) :=
  PhiIO.decodeIO Native.algebra g order id id

def job (g : PhiIO.Geometry) (row : Nat) : Invocation Native.Term :=
  { PhiIO.launch Native.algebra g row with
    kernel := Candidates.phiSharedBase g.H g.S g.D g.Q g.BD }

def variant (g : PhiIO.Geometry) (order : List Nat) :
    ProgramIO.Variant (io g order) where
  schedule := fun _ => order.map (job g)
  observe := (io g order).observe

end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.IOPair
