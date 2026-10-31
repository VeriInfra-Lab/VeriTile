import bench.tritonbench_g.relu_strided_buffer.Kernels
import VeriTile.Triton
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WholeSource
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WordHost

/-! Exact selected-FP32 source value semantics. This source only selects a raw
input word or +0; there is no arithmetic roundoff. The literal-zero conversion
is instantiated explicitly, so ZeroLiteralLaw is discharged rather than assumed.
Software instruction semantics and ordinary typed-memory representation are
separate from any claim of a native GPU measurement. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton FP.Structural HostPlan SignedView SignedGrid
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Exact implementation of the only untyped literal and cross-tag cast this
source uses: mathematical zero is represented by the positive FP32 zero word.
All other operations stay explicit in rest and are irrelevant to this proof. -/
def zeroLiterals (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  { rest with
    literal := fun p d x => if p = none ∧ d = .real ∧ x = 0 then
      FP.Value.zero .fp32 else rest.literal p d x
    cast := fun p src dst x =>
      if p = none ∧ src = .real ∧ dst = .fp32 ∧ x = FP.Value.zero .fp32 then
        FP.Value.zero .fp32 else rest.cast p src dst x }

theorem zero_literal_exact (rest : Algebra (FP.Value .fp32)) :
    WordTile.ZeroLiteralLaw (zeroLiterals rest) := by
  simp [WordTile.ZeroLiteralLaw, zeroLiterals]

def fp32Backend (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  WordTile.backend (zeroLiterals rest) .ieeeValue

/-- Closed bit-classification form of the strict ordered selection. This is
proved for every FP32 word; no finite/normal-only assumption is used. -/
theorem fp32_result_classification (x : FP.Value .fp32) :
    WordTile.wordRelu x =
      if x.isNaN || x.isZero || x.sign then FP.Value.zero .fp32 else x := by
  have zn : (FP.Value.zero .fp32).isNaN = Bool.false := by decide
  have zz : (FP.Value.zero .fp32).isZero = Bool.true := by decide
  have zs : (FP.Value.zero .fp32).sign = Bool.false := by decide
  have zb : (FP.Value.zero .fp32).bits.toNat = 0 := by decide
  by_cases nan : x.isNaN = Bool.true
  · simp [WordTile.wordRelu, WordTile.orderedLt,
      ReluTritonKernel.SoftwareFP32.orderedLe, nan, zn]
  have nan' : x.isNaN = Bool.false := Bool.eq_false_iff.mpr nan
  by_cases zero : x.isZero = Bool.true
  · simp [WordTile.wordRelu, WordTile.orderedLt,
      ReluTritonKernel.SoftwareFP32.orderedLe, nan', zero, zn, zz]
  have zero' : x.isZero = Bool.false := Bool.eq_false_iff.mpr zero
  have nonzero : x.bits.toNat ≠ 0 := by
    intro rawZero
    have zeroTrue : x.isZero = Bool.true := by
      simp [FP.Value.isZero, FP.Value.exponent, FP.Value.fraction, FP.Value.raw, rawZero]
    exact zero zeroTrue
  cases sign : x.sign <;>
    simp [WordTile.wordRelu, WordTile.orderedLt,
      ReluTritonKernel.SoftwareFP32.orderedLe, nan', zero', sign, zn, zz, zs, zb, nonzero]

/-- Independent per-element FP32 specification. Ordered strict comparison
rejects NaNs and both zero signs; the selected positive word is unchanged. -/
def fp32Result (xs : Nat → FP.Value .fp32) (i : Nat) : FP.Value .fp32 :=
  let x := xs i
  if x.isNaN || x.isZero || x.sign then FP.Value.zero .fp32 else x

/-- Little-endian byte observation of a selected FP32 cell. It is ordinary
memory representation, not an additional scalar kernel implementation. -/
def wordByte (x : FP.Value .fp32) (byte : Nat) : BitVec 8 :=
  x.bits.extractLsb' (8 * byte) 8

def memoryByte (s : State (FP.Value .fp32)) (r : RegionName) (a byte : Nat) : BitVec 8 :=
  wordByte ((s.mem r a).read .fp32) byte

theorem output_bytes (s : State (FP.Value .fp32)) (r : RegionName) (a : Nat)
    (x : FP.Value .fp32) (h : s.mem r a = .mk .fp32 x) (byte : Nat) :
    memoryByte s r a byte = wordByte x byte := by
  unfold memoryByte
  rw [h]
  rfl

theorem frame_bytes (s t : State (FP.Value .fp32)) (r : RegionName) (a : Nat)
    (h : t.mem r a = s.mem r a) (byte : Nat) :
    memoryByte t r a byte = memoryByte s r a byte := by
  simp [memoryByte, h]

#guard_msgs (drop info) in
#specNonCircular fp32Result avoiding [VeriTile.Bench.Repairs.ReluStridedBuffer.kernel]
#axiomsClean zero_literal_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
