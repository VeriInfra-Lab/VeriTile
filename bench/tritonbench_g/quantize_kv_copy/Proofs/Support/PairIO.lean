import bench.tritonbench_g.quantize_kv_copy.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Complete address-early candidate views reuse sourceIO windows and metadata. -/
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Pair
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
set_option autoImplicit true

def invocation (op : Ops α) (p : Parameters) (pid : Nat) : Invocation α :=
  { PublicIO.invocation op p pid with
    kernel := KernelSyntax.addressEarly p.groups p.block p.heads }

def variant (op : Ops α) (p : Parameters) (n : Nat) (dest : Nat → Nat) :
    ProgramIO.Variant (PublicIO.sourceIO op p n dest) where
  schedule := fun _ => (List.range n).map (invocation op p)
  observe := (PublicIO.sourceIO op p n dest).observe

def original (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) :=
  ProgramIO.Implementation.original
    (PublicIO.sourceIO (KernelModels.nativeOps instructions tree) p n dest) (PublicIO.Metadata dest)

def addressEarly (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) :=
  ProgramIO.Implementation.ofVariant
    (variant (KernelModels.nativeOps instructions tree) p n dest) (PublicIO.Metadata dest)

local instance : Inhabited KernelModels.Word := ⟨KernelModels.zero⟩

end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Pair
