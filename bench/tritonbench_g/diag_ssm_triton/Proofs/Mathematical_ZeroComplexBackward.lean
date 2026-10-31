import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroBackward
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroComplexBackward
open VeriTile.Triton TypedSource
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def window (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (region : RegionName) (component o : Nat) : Prop :=
  ∃ j : Fin block, u.pids 0*block+j.val < batch*dim ∧
    o = A.addr region ((u.pids 0*block+j.val)*2+component)

def written (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (gs gl r : RegionName) (o : Nat) : Prop :=
  r = A.flat ∧ (window A u batch dim block gs 0 o ∨ window A u batch dim block gs 1 o ∨
    window A u batch dim block gl 0 o ∨ window A u batch dim block gl 1 o)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroComplexBackward
