import bench.repairs.tritonbench_g.parallel_attention.Proofs.Torch
import bench.repairs.tritonbench_g.parallel_attention.Proofs.PublicReal

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
noncomputable section
open HostAPI

/-- Actual caller Real observations use the same absolute aligned element
addresses as the byte endpoint; aliasing therefore has the same meaning. -/
def realObserve (h : Nat → ℝ) : Memory ℝ := fun a=>h (4*a)

def invokeReal (c : Call) (d : Domain c) (profile : Syntax)
    (r : Public.Run c.shape (forwardBases (inputs c)) 128 forwardKinds)
    (t : ForwardTensors) (eps : ℝ) (den dst : Nat) (h : Nat → ℝ) : Result (Memory ℝ) :=
  ⟨returned c t,[c.q.identity,c.k.identity,c.v.identity],c.q.tensor.device.index,c.callerDevice,
    Machine.launches c.shape 128,
    (logicalForward c d profile r eps den dst realOps (realObserve h)).api.memory⟩

end
end VeriTile.Bench.Repairs.ParallelAttention.Checked
