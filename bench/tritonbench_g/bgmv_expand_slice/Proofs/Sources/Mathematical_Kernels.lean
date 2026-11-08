import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit
/-! Pure storage tags and the host cast choice shared by the repaired call
descriptor. Executable source lives in SurfaceSource and Kernels. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 2400000
inductive InputStorage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

/-- Weight storage types accepted by the host wrapper. -/
inductive WeightStorage where
  | fp16 | bf16
  deriving DecidableEq

abbrev KernelSource := FloatDType → RegionName → RegionName → RegionName → Nat → Nat →
  Region .int → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Bool → Bool → ComputeKernel

/-- The host's CAST_TYPE selection; the sources also support explicit choices. -/
def hostCast : InputStorage → Bool
  | .fp32 => Bool.true
  | _ => Bool.false

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource
