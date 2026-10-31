import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
import VeriTile.Triton.Semantics.Step
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroScatter
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroBackward
open VeriTile.Triton TypedSource
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 600000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def window (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (region : RegionName) (o : Nat) : Prop :=
  ∃ j : Fin block, u.pids 0 * block + j.val < batch * dim ∧
    o = A.addr region (u.pids 0 * block + j.val)

def written (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (gs gl r : RegionName) (o : Nat) : Prop :=
  r = A.flat ∧ (window A u batch dim block gs o ∨ window A u batch dim block gl o)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroBackward
