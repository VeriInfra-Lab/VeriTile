import bench.tritonbench_g.flash_decode2_llama.Proofs.NativeIO
import bench.tritonbench_g.flash_decode2_llama.Kernels
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.IOPair
open VeriTile Triton FP.Structural FlashDecode2
noncomputable section

def io (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat) :=
  IO.decodeIO Native.algebra g order Native.Term.bits (NativeIO.observe p g)

def job (g : KernelInterface.Geometry) (row : Nat) : Invocation Native.Term :=
  { IO.launch Native.algebra g row with
    kernel := Candidates.llamaSharedBase g.H g.S g.D g.Q g.D }

def variant (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat) :
    ProgramIO.Variant (io p g order) where
  schedule := fun _ => order.map (job g)
  observe := (io p g order).observe

end
end VeriTile.Bench.Repairs.FlashDecode2Llama.IOPair
