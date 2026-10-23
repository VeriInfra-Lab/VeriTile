import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.IOData
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
open Classical
set_option maxHeartbeats 2500000

@[simp] theorem decode_encode (c : Call) (j : Coordinate c) : decode c (encode c j)=j := Equiv.symm_apply_apply _ _

structure Metadata (c : Call) (r : Regions) (s : BlockState) : Prop where
  start : ∀ b,b<c.batches → (Program.inputs s r).start b=(c.start b : Int)
  length : ∀ b,b<c.batches → (Program.inputs s r).sequence b=(c.length b : Int)

theorem metadata (c : Call) (r : Regions) (xs : Values c) (hd : domain c xs)
    (s : BlockState) (hin : (io c r).InputBinding s xs) : Metadata c r s := by
  constructor
  · intro b hb
    exact (hin 3 ⟨b,hb⟩ trivial).trans (hd.1 ⟨b,hb⟩)
  · intro b hb
    exact (hin 4 ⟨b,hb⟩ trivial).trans (hd.2 ⟨b,hb⟩)

#axiomsClean metadata
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
