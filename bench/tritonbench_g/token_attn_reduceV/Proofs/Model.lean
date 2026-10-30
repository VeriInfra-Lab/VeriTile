import bench.tritonbench_g.token_attn_reduceV.Sources.Source
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
namespace VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section

/-- Total fp32 reduction availability. The full array, ax, precision and layout
are arguments; no algebraic sum law or padding identity is assumed. -/
structure SumPrimitive (α : Type) where
  apply : Option ComputeDType → {shape : TileShape} → (ax : Fin shape.length) →
    (keep : Bool) → Values α .fp32 shape → Values α .fp32 (TileShape.reduceShape shape ax keep)

def withSum {α : Type} (M : Algebra α) (R : SumPrimitive α) : Algebra α :=
  { M with reduceSumTyped := fun pr d {_} ax keep xs =>
      match d with
      | .fp32 => some (R.apply pr ax keep xs)
      | .real => M.reduceSumTyped pr .real ax keep xs
      | .fp16 => M.reduceSumTyped pr .fp16 ax keep xs
      | .bf16 => M.reduceSumTyped pr .bf16 ax keep xs
      | .f8e4 => M.reduceSumTyped pr .f8e4 ax keep xs
      | .f8e5 => M.reduceSumTyped pr .f8e5 ax keep xs }

def count {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.length) (s.pids 0*p.lengthStride)).read .nat

def startIndex {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.start) (s.pids 0*p.startStride)).read .nat

def requestIndex {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.request) (s.pids 0*p.requestStride)).read .nat

def probOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (n : Nat) :=
  s.pids 1 * p.probHeadStride + (startIndex p s + n) * p.probTokenStride

def gatherOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (n : Nat) :=
  requestIndex p s * p.requestBatchStride + n * p.requestTokenStride

def valueOffset {α : Type} (p : Parameters) (s : State α) (d : Nat) :=
  s.pids 1 / p.kvGroup * p.valueHeadStride + d * p.valueDimStride

def padding {α : Type} (M : Algebra α) : α :=
  M.cast none .real .fp32 (M.literal none .real 0)

def probability {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (start n : Nat) : α :=
  if start+n < count p s then (s.mem p.prob (probOffset p s n+start*p.probTokenStride)).read .fp32
  else padding M

def gather {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start n : Nat) : Nat :=
  if start+n < count p s then
    (s.mem (Region.cast p.tokens) (gatherOffset p s n+start*p.requestTokenStride)).read .nat
  else 0

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (start n d : Nat) : α :=
  if start+n < count p s then
    (s.mem p.value (valueOffset p s d+gather p s start n*p.valueBatchStride)).read .fp32
  else padding M

/-- All BLOCK_N products are evaluated, including padded lanes. In particular
there is no cancellation of zero products or zero block contributions. -/
def products {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (start : Nat) : TileIndex [p.block,p.width] → α := fun i =>
  M.binary (some .fp32) .fp32 .mul (probability M p s start i.1.val)
    (value M p s start i.1.val i.2.1.val)

def update {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (start : Nat) (previous : TileIndex [p.width] → α) :
    TileIndex [p.width] → α :=
  let blockValue := R.apply (some .fp32) ⟨0, by simp⟩ Bool.false (products M p s start)
  fun i => M.binary (some .fp32) .real .add (previous i)
    (M.cast (some .fp32) .fp32 .real (blockValue i))

def initial {α : Type} (M : Algebra α) (p : Parameters) : TileIndex [p.width] → α :=
  fun _ => M.literal (some .fp32) .real 0

def accumulated {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : TileIndex [p.width] → α :=
  foldRange (update M R p s) 0 (count p s) p.block (initial M p)

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) : α :=
  M.cast none .real .fp32 (accumulated M R p s i)

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) :=
  s.pids 0*p.outputBatchStride+s.pids 1*p.outputHeadStride+i.1.val*p.outputDimStride

def reference {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.write p.output (outputOffset p s i) (.mk .fp32 (outputValues M R p s i))) s

end
end VeriTile.Bench.Repairs.TokenAttnReduceV
