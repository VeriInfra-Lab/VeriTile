import bench.tritonbench_g.fast_ce_loss.Kernels
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit

/-! The three complete JIT bodies at their already prepared argument pointers.
Each region starts at the actual data pointer, including its storage offset.
Row strides below are the API observations passed to the kernel. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section

structure Parameters where
  logits : RegionName
  labels : Region .int
  loss : RegionName
  lse : RegionName
  upstream : RegionName
  rows : Nat
  vocab : Nat
  chunks : Nat
  block : Nat
  logitsStride : Nat
  upstreamStride : Nat
  softcap : ℝ
  scaling : ℝ
  hasSoftcap : Bool
  hasScaling : Bool

def forward (p : Parameters) : ComputeKernel :=
  TypedSource.forward p.hasSoftcap p.hasScaling p.logits p.loss p.lse p.labels
    p.vocab p.logitsStride p.block p.softcap p.scaling

def chunked (p : Parameters) : ComputeKernel :=
  TypedSource.chunked p.hasSoftcap p.hasScaling p.logits p.loss p.lse p.labels
    p.vocab p.chunks p.logitsStride p.block p.softcap p.scaling

def backward (p : Parameters) : ComputeKernel :=
  TypedSource.backward p.hasSoftcap p.hasScaling p.logits p.upstream p.lse p.labels
    p.vocab p.logitsStride p.upstreamStride p.block p.softcap p.scaling

def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

def label {α : Type} [Inhabited α] (p : Parameters) (initial : State α) : Int :=
  (initial.mem (Region.cast p.labels) (initial.pids 0)).read .int

def forwardFrontLength (p : Parameters) : Nat :=
  12 + (if p.hasScaling then 1 else 0) + (if p.hasSoftcap then 1 else 0)

def forwardFront (p : Parameters) := (forward p).surfaceBody.take (forwardFrontLength p)
def forwardTail (p : Parameters) := (forward p).surfaceBody.drop (forwardFrontLength p)

theorem forward_split (p : Parameters) :
    (forward p).surfaceBody=forwardFront p++forwardTail p :=
  (List.take_append_drop (forwardFrontLength p) _).symm

end
end VeriTile.Bench.Repairs.FastCeLoss.KernelSource
