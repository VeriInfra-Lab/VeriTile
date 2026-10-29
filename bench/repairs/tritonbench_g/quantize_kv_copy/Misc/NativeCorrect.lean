import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.IOGrid

/-! Source-to-native observation for the actual grouped quantization source.
The five native instruction ports and eight-lane reduction tree retain their
existing precision/conversion boundaries. No second kernel, floating rewrite,
finite-input gate or new numerical experiment assumption is introduced. -/
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.FPEquiv
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
local instance : Inhabited KernelModels.Word := ⟨KernelModels.zero⟩

/-- The complete typed source and both observable outputs share IO.lean's
metadata-bound windows and actual token/head launch schedule. -/
def nativeIO (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) : ProgramIO KernelModels.Word KernelModels.Word :=
  PublicIO.sourceIO (KernelModels.nativeOps instructions tree) p n dest

/-- Actual source execution safely produces every ordered native code and
FP32 scale, with the entire exterior cell frame. All FP32 words are admitted;
exceptional INT8 conversion follows the explicit primitive profile. -/
theorem native (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) (covers : p.groups ≤ p.block)
    (unique : ∀ a b, a<n → b<n →
      KernelGridResult.target p dest a = KernelGridResult.target p dest b → a=b) :
    (nativeIO instructions tree p n dest).ImplementsOn (PublicIO.Metadata dest)
      (fun values => PublicIO.result
        (fun pid g c => (KernelModels.nativeCode instructions tree (PublicIO.elements values pid g) c).toInt)
        (fun pid g => KernelModels.nativeScale instructions tree (PublicIO.elements values pid g))) := by
  apply PublicIO.implements
  intro values s reads
  exact KernelRaw.native instructions tree p n dest covers unique (PublicIO.elements values) s reads

#specNonCircular KernelModels.nativeCode avoiding [Kernel.kernel, FP.Structural.exec, KernelGrid.run]
#specNonCircular KernelModels.nativeScale avoiding [Kernel.kernel, FP.Structural.exec, KernelGrid.run]
#axiomsClean native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.FPEquiv
