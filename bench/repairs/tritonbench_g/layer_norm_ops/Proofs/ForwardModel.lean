import bench.repairs.tritonbench_g.layer_norm_ops.Kernels
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection

namespace VeriTile.Bench.Repairs.LayerNormOps.Forward
open VeriTile.Triton FP.Structural
noncomputable section

def binary {α : Type} (M : Algebra α) (op : Binary) := M.binary (some .fp32) .real op
def unhintedBinary {α : Type} (M : Algebra α) (op : Binary) := M.binary none .real op
def literal {α : Type} (M : Algebra α) (v : ℝ) := M.literal (some .fp32) .real v
def padding {α : Type} (M : Algebra α) := M.cast (some .fp32) .real .fp32 (literal M 0)
def cast {α : Type} (M : Algebra α) := M.cast (some .fp32) .fp32 .real
def stored {α : Type} (M : Algebra α) (v : α) : Cell α := .mk .fp32 (M.cast none .real .fp32 v)

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (region : RegionName) (base : Nat) (i : TileIndex [p.block]) : α :=
  cast M (if i.1.val < p.columns then (s.mem region (base+i.1.val)).read .fp32 else padding M)

def unspecified {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (region : RegionName) (i : TileIndex [p.block]) : α :=
  cast M (if i.1.val < p.columns then (s.mem region i.1.val).read .fp32
    else s.undef .fp32 region i.1.val)

def z {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (i : TileIndex [p.block]) : α :=
  let x := loaded M p s p.x (s.pids 0*p.xStride) i
  if p.hasResidual then binary M .add x (loaded M p s p.residual (s.pids 0*p.residualStride) i)
  else x

def average {α : Type} (M : Algebra α) (p : ForwardParameters)
    (values : TileIndex [p.block] → α) : α :=
  binary M .div (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false values PUnit.unit)
    (literal M (p.columns : ℝ))

def meanValue {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  average M p (z M p s)

def centered {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (i : TileIndex [p.block]) : α :=
  if p.rms then z M p s i else binary M .sub (z M p s i) (meanValue M p s)

def varianceValue {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  average M p (fun i =>
    let xbar := if i.1.val < p.columns then centered M p s i else literal M 0
    binary M .mul xbar xbar)

def rstdValue {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  binary M .div (literal M 1)
    (M.unary (some .fp32) .sqrt (binary M .add (varianceValue M p s) (literal M p.epsilon)))

def scatter {α : Type} (p : ForwardParameters) (pid : Nat) (region : RegionName) (stride : Nat)
    (values : TileIndex [p.block] → Cell α) (s : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val < p.columns then
    t.write region (pid*stride+i.1.val) (values i) else t) s

def residualState {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  if p.storeResidual then scatter p (s.pids 0) p.residualOut p.residualOutStride
    (fun i => stored M (z M p s i)) s else s

def statisticsState {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  let t := residualState M p s
  let u := if p.rms then t else t.write p.mean (s.pids 0) (stored M (meanValue M p s))
  u.write p.rstd (s.pids 0) (stored M (rstdValue M p s))

def yValue {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (i : TileIndex [p.block]) :=
  let t := statisticsState M p s
  let base := if p.rms then z M p s i else unhintedBinary M .sub (z M p s i) (meanValue M p s)
  let h := unhintedBinary M .mul base (rstdValue M p s)
  let y := unhintedBinary M .mul h (unspecified M p t p.weight i)
  if p.hasBias then unhintedBinary M .add y (unspecified M p t p.bias i) else y

/-- Explicit initial-state recurrence, including the source order of early
residual/statistic writes and later parameter loads. Public freshness lemmas
remove these possible aliases; the execution theorem itself assumes none. -/
def reference {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters) (s : State α) :=
  scatter p (s.pids 0) p.y p.yStride (fun i => stored M (yValue M p s i)) (statisticsState M p s)

end
end VeriTile.Bench.Repairs.LayerNormOps.Forward
