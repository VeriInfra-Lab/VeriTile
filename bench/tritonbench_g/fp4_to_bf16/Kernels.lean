import bench.tritonbench_g.fp4_to_bf16.Sources.Mathematical_Kernels

/-! Separate repair of the pinned Python (SHA256
8571cb56ce932af6a7dd68e4047ee0863984812ac303d0d2959fc544a65e9958).
The conversion body is unchanged. Indices/counts are widened before arithmetic;
`Proofs.Finite` proves their refinement to the reused mathematical-index body.
The uint8 input denotes packed bytes at the passed data pointer, high nibble
first. Precision is signed32 payload → FP32 bitcast → BF16, including signed 0.
This is a source/primitive-semantics model, not a compiler-correctness theorem. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
namespace Kernels
abbrev body := FPValue.unscaled
/-- The shared readable typed conversion source, with mathematical indices.
`Finite` proves each widened native count/index/mask/address operation exact;
`Launch` binds the actual native count and program-id ABI to this source.
The conversion body, masks, stores and 512-byte launch are unchanged. -/
def source (p : FPValue.Parameters) := body p.input p.output p.count p.block
abbrev index := BitVec 64
/-- The cast precedes the multiply; both i32 and u32 count specializations
therefore denote the same nonnegative mathematical count after widening. -/
def count (n : Nat) : index := BitVec.ofNat 64 n
def outputCount (n : Nat) : index := count n * BitVec.ofNat 64 2
def inputIndex (pid lane : Nat) : index :=
  BitVec.ofNat 64 pid * BitVec.ofNat 64 512 + BitVec.ofNat 64 lane
def outputIndex (pid lane : Nat) : index :=
  BitVec.ofNat 64 pid * BitVec.ofNat 64 1024 + BitVec.ofNat 64 lane
def inputMask (n pid lane : Nat) : Bool := (inputIndex pid lane).slt (count n)
def outputMask (n pid lane : Nat) : Bool := (outputIndex pid lane).slt (outputCount n)
def pointer (base stride : Nat) (offset : index) : BitVec 64 :=
  BitVec.ofNat 64 base + BitVec.ofNat 64 stride * offset
end Kernels
end VeriTile.Bench.Repairs.Fp4ToBf16
