import bench.repairs.tritonbench_g.triton_attention.Whole001.Correct
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.PreprocessReduction
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001

def witnessShape : Shape := ⟨2,3,3,16,by decide,by decide,by decide,by decide,by decide,by decide⟩
def witnessCoord : Coord witnessShape := (⟨4,by decide⟩,⟨270,by decide⟩,⟨15,by decide⟩)

theorem nontrivial_batch_head_row : offset witnessShape witnessCoord = 28911 ∧
    witnessCoord.2.1.val / 128 = 2 ∧ witnessCoord.2.1.val % 128 = 14 := by decide

def zeroInstructions : Native.Instructions :=
  ⟨fun _ _ => 0,fun _ _ => 0,fun _ _ => 0,fun _ _ _ => 0,fun _ => 0,
   fun _ _ => 0,fun _ => 0,fun _ _ => 0,fun _ => 0,fun _ => 0,fun _ _ _ _ _ => 0⟩
def changedInstructions : Native.Instructions := { zeroInstructions with add := fun _ _ => 1 }

theorem numerical_not_automatic : ¬Native.LocalConformance changedInstructions zeroInstructions := by
  intro h
  have bad := h.add 0 0
  change (1 : BitVec 32) = 0 at bad
  contradiction

def witnessArguments : Host.Arguments :=
  ⟨Host.tensor witnessShape 0 .q,Host.tensor witnessShape 0 .k,
   Host.tensor witnessShape 0 .v,0x3f000000⟩
def witnessGradient : Host.Tensor :=
  { Host.tensor witnessShape 0 .g with strides := [0,0,0,0] }

theorem admitted_domain_nonempty : Host.InputChecks witnessShape witnessArguments ∧
    Host.GradientChecks witnessShape witnessArguments witnessGradient := by
  unfold Host.InputChecks Host.GradientChecks
  decide

#axiomsClean nontrivial_batch_head_row
#axiomsClean numerical_not_automatic
#axiomsClean admitted_domain_nonempty
end VeriTile.Bench.Repairs.TritonAttention.Whole001
