import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Kernels
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InterModel
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton
noncomputable section
def fwdParts (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (forward p).surfaceBody "i").getD ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def interForwardParts (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (inter p).surfaceBody "i").getD ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def interReverseParts (p : Parameters) : Source.IntRange :=
  (Source.intRange? (interForwardParts p).after "__rev_i").getD ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩
def interRUpdateParts (p : Parameters) : Source.IntRange :=
  (Source.intRange? (interReverseParts p).body "__rev_j").getD ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩
def interRDeltaParts (p : Parameters) : Source.IntRange :=
  (Source.intRange? (interRUpdateParts p).after "__rev_j").getD ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩
def reverseStart (n : Nat) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt n) (.constInt 1))
def reverseIndex (name : RegName) : ComputeStmt :=
  .assign .int [] name (.alg (.sub .int .nil (.constInt 0) (.ref .int [] ("__rev_" ++ name))))
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
