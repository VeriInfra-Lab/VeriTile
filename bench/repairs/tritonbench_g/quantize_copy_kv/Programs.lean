import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.NativeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
open KernelNativeProfile (Word)
local instance wordInhabited : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Full original schedule, retaining source reduction order, div.full,
FP16 rounding/promotion and total PTX signed-byte conversion on all words. -/
def originalIO (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) : ProgramIO Word Word :=
  Interface.programIO (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal
    p count dest Bool.false

/-- Same channels and footprint; the candidate changes only an integer
address product in every invocation of the complete token schedule. -/
def commuted (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) :
    ProgramIO.Variant (originalIO a rest p count dest) :=
  Interface.candidate (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal p count dest

/-- Independent native code/scale target, including exceptional operands. -/
def native (a : KernelNativeProfile.Arithmetic) (p : Parameters) (count : Nat)
    (values : Interface.Inputs Word p count) : Interface.Outputs Word p count :=
  Interface.result (RawKernel.nativeCode a p (Interface.payload values))
    (RawKernel.nativeScale a p (Interface.payload values))

/-- The complete original token schedule on the existing logical metadata domain. -/
def original (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) :=
  ProgramIO.Implementation.original (originalIO a rest p count dest)
    (Interface.metadataMatches p count dest)

/-- The already proved integer-address candidate, on exactly the same IO. -/
def indexCommuted (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) :=
  ProgramIO.Implementation.ofVariant (commuted a rest p count dest)
    (Interface.metadataMatches p count dest)

end
end VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
