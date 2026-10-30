import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Interface
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.RawNative

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

theorem native_implements (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool)
    (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    (programIO (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal
      p count dest version).ImplementsOn (metadataMatches p count dest)
      (fun values => result (RawKernel.nativeCode a p (payload values))
        (RawKernel.nativeScale a p (payload values))) := by
  apply implements _ _ _ p count dest version width
  intro values s domain binding
  exact RawNative.kernel_equiv a rest p count dest width covered unique (payload values) s
    (metadata_binding _ _ _ p count dest version values domain s binding)
    (payload_binding _ _ _ p count dest version values s binding)

theorem native_equivalent (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    (programIO (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal
      p count dest Bool.false).EquivalentOn
        (candidate (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal p count dest)
        (metadataMatches p count dest) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (native_implements a rest p count dest Bool.false width covered unique)
    (native_implements a rest p count dest Bool.true width covered unique)

#axiomsClean native_implements
#axiomsClean native_equivalent
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
