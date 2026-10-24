import bench.tritonbench_g.layer_norm_liger.Proofs.FPTypedSource
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section

def padding {α : Type} (M : Algebra α) : α := M.cast none .real .fp32 (M.literal none .real 0)
def input {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (region : RegionName) (base : Nat) : TileIndex [p.block] → α := fun i =>
  if i.1.val < p.columns then (s.mem region (base+i.1.val)).read .fp32 else padding M

def xValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  input M p s p.x (s.pids 0*p.xStride)
def wValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  input M p s p.weight 0
def bValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  input M p s p.bias 0

def denominator {α : Type} (M : Algebra α) (p : Parameters) : α :=
  M.cast (some .fp32) .real .real (M.literal (some .fp32) .real (p.columns : ℝ))
def meanValue {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .fp32 .real
      (R.apply (some .fp32) ⟨0, by simp⟩ Bool.false (xValues M p s) PUnit.unit)) (denominator M p)
def centered {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) : α :=
  M.binary (some .fp32) .real .sub
    (M.cast (some .fp32) .fp32 .real (xValues M p s i)) (meanValue M R p s)
/-- Full padded block: unlike the streaming implementation there is no mask
between subtracting the mean and squaring. Every padded (0-mean)^2 is retained. -/
def varianceValue {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div
    (M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false
      (fun i : TileIndex [p.block] => M.binary (some .fp32) .real .mul
        (centered M R p s i) (centered M R p s i)) PUnit.unit) (denominator M p)
def rstdValue {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : α :=
  M.unary (some .fp32) .rsqrt (M.binary (some .fp32) .real .add
    (varianceValue M R p s) (M.literal (some .fp32) .real p.epsilon))
def yValue {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul (centered M R p s i) (rstdValue M R p s))
      (M.cast (some .fp32) .fp32 .real (wValues M p s i)))
    (M.cast (some .fp32) .fp32 .real (bValues M p s i))
def stored {α : Type} (M : Algebra α) (v : α) : Cell α :=
  .mk .fp32 (M.cast none .real .fp32 v)
def meanOffset {α : Type} (p : Parameters) (s : State α) := s.pids 0*p.meanStride
def rstdOffset {α : Type} (p : Parameters) (s : State α) := s.pids 0*p.rstdStride
def outputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.block]) :=
  s.pids 0*p.yStride+i.1.val

def reference {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) : State α :=
  let t := (s.write p.mean (meanOffset p s) (stored M (meanValue M R p s))).write
    p.rstd (rstdOffset p s) (stored M (rstdValue M R p s))
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val < p.columns then
    t.write p.y (outputOffset p s i) (stored M (yValue M R p s i)) else t) t

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r ≠ p.mean ∨ o ≠ meanOffset p s) ∧ (r ≠ p.rstd ∨ o ≠ rstdOffset p s) ∧
  (r ≠ p.y ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o ≠ outputOffset p s i)
end
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
