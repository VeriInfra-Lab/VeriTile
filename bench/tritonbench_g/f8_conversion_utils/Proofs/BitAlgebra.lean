import bench.tritonbench_g.f8_conversion_utils.Proofs.SourceStorage

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source.BitAlgebra
open VeriTile.Triton FP.Structural Bytes
noncomputable section
set_option maxRecDepth 24000

/-- A scalar register stores its bits, including every NaN payload and sign.
This carrier is not a Real interpretation and does not collapse bit patterns. -/
abbrev Scalar := Nat → BitVec 8

def pack (word : Word width) : Scalar := fun i => if bound : i<width then word ⟨i,bound⟩ else 0
def unpack (width : Nat) (value : Scalar) : Word width := fun k => value k.val

theorem unpack_pack (word : Word width) : unpack width (pack word)=word := by
  funext k
  simp [unpack,pack,k.isLt]

/-- The only floating operation reached by these original kernels is their
one configured scalar cast. All source integer index arithmetic remains in
Structural's ordinary integer semantics. Unreached floating operations are
irrelevant to the checked whole-source execution theorem. FP32 storage loads
are explicit, bit-preserving register loads in this carrier. -/
def algebra (profile : Source.Profile)
    (physical : Word profile.inputWidth → Word profile.outputWidth) : Algebra Scalar where
  literal := fun _ _ _ => fun _ => 0
  negInf := fun _ => 0
  binary := fun _ _ _ _ _ => fun _ => 0
  unary := fun _ _ _ => fun _ => 0
  cast := fun _ _ _ bits => pack (physical (unpack profile.inputWidth bits))
  fromNat := fun _ _ => fun _ => 0
  fromInt := fun _ _ => fun _ => 0
  fp32Bits := fun _ => fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => fun _ => 0

/-- The source/native scalar binding is constructed from the local physical
word conversion. Existing output memory is not converted or canonicalized. -/
def binding (profile : Source.Profile)
    (physical : Word profile.inputWidth → Word profile.outputWidth) :
    Source.ScalarBinding (algebra profile physical) profile physical where
  loadWord := pack
  storeWord := unpack profile.outputWidth
  conversion := by
    intro word
    cases profile <;> simp [Source.cast,algebra,Algebra.withDefaultPrecision,unpack_pack]

def outputStorage (profile : Source.Profile)
    (physical : Word profile.inputWidth → Word profile.outputWidth) :
    Source.OutputStorage profile (binding profile physical).storeWord :=
  ⟨pack,unpack_pack⟩

#axiomsClean unpack_pack
#axiomsClean binding
#axiomsClean outputStorage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source.BitAlgebra
