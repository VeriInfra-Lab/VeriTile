import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEIO

/-! Actual whole-grid source equivalence for the selected FP32 kernel and
its per-iteration key-tail-mask CSE. The exact native source graph is retained,
so every supplied owner-specific native interpretation has equal outputs. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.FPEquiv
open VeriTile Triton FP.Structural Spec
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

def original (c : Call) :=
  ArrayProgramIO.Implementation.original (NativeInterface.programIO c) (fun _ => True)

def candidate (c : Call) :=
  ArrayProgramIO.Implementation.ofVariant (io:=NativeInterface.programIO c)
    {schedule := (CSEInterface.programIO c).schedule,
     observe := (CSEInterface.programIO c).observe} (fun _ => True)

/-- Both complete actual ordinary grids execute safely, every active padded
FP32 O/M/L cell agrees, and both complete exterior frames hold. Six arbitrary
input arrays and the raw FP32 scale slot are unchanged; no numerical law or
finite-word condition is added. -/
specification key_mask_equiv (c : Call) (positive : 0<c.BN) :
    original c ≡[[]] candidate c := by
  apply ArrayProgramIO.Implementation.of_equivalent
  exact ArrayProgramIO.EquivalentOn.of_implementsOn
    (NativeInterface.implements c positive) (CSEInterface.implements c positive)

#axiomsClean key_mask_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.FPEquiv
