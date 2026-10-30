import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeFP
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Backend
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Compiler-observed ordered scan graph. Fused nodes retain the original
product operands through a graph carrier; they cannot act on rounded products. -/
inductive SumTree (n : Nat) where
  | leaf (index : Fin n)
  | add (left right : SumTree n)
  | fused (index : Fin n) (addend : SumTree n)

def SumTree.leaves {n : Nat} : SumTree n → List Nat
  | .leaf i => [i.val]
  | .add l r => l.leaves ++ r.leaves
  | .fused i r => [i.val] ++ r.leaves

def SumTree.eval {n : Nat} (add fused : α → α → α) (x : Fin n → α) : SumTree n → α
  | .leaf i => x i
  | .add l r => add (l.eval add fused x) (r.eval add fused x)
  | .fused i r => fused (x i) (r.eval add fused x)

structure PrefixPlan (h n : Nat) where
  tree : Fin h → Fin n → SumTree n
  leaves : ∀ i j, (tree i j).leaves.Perm (List.range (j.val+1))

/-- Only local numerical primitives are supplied: comparisons and row scan.
The scan law expands one invocation into an explicit ordered graph. Fused
ports are instantiated by graph constructors, not by an FP32-product oracle.
It cannot supply final kernel output or a successful kernel run. -/
structure Primitives (α : Type) (c : Whole.Host) where
  model : Algebra α
  less : α → α → Bool
  lessEqual : α → α → Bool
  lt_support : model.compareLt (some .fp32) .real = some less
  le_support : model.compareLe (some .fp32) .real = some lessEqual
  fused : α → α → α
  plan : PrefixPlan c.BH c.BC
  scan_support : ∀ x : Values α .real [c.BH,c.BC],
    model.scan (some .fp32) (shape := [c.BH,c.BC]) .sum (1 : Fin 2) .forward x =
      some (fun i => (plan.tree i.1 i.2.1).eval (model.binary (some .fp32) .real .add) fused
        (fun k => x (i.1,k,PUnit.unit)))

@[simp] theorem scatter_regs {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write region (off i) (value i) else t) s).regs = s.regs := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi,if_true,State.write] using ih (s.write region (off i) (value i))
    · simpa only [hi,if_false] using ih s

def sourceRun {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (s : State α) :
    {v : State α // run (engine p.model)
      (WholeFP.typed32 c r Bool.false hasBias useSoftplus lower upper).surfaceBody
      (WholeFP.tileState l s) = some v} := by
  cases hasBias <;> cases useSoftplus <;> cases upper
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 900000) [WholeFP.typed32,Program.kernel,source,finiteSource,infiniteSource,
      ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,natLt,natLe,numericLt,numericLe,WholeFP.tileState,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,Whole.Host.layout,scatter_regs,p.lt_support,p.le_support,p.scan_support]
    try rfl

/-- Independent scalar preparation graph. Storage/load casts, both boolean
branches, cutoff softplus, finite/default clamp and the final remask remain. -/
def prepare {α : Type} (c : Whole.Host) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (active : Bool) (x bias : α) : α :=
  let m := p.model
  let x := m.cast (some .fp32) .fp32 .real x
  let bias := m.cast (some .fp32) .real .real bias
  let x := if hasBias then m.binary (some .fp32) .real .add x bias else x
  let x := if useSoftplus then
    if p.lessEqual x (m.literal (some .fp32) .real 20.0) then
      m.unary (some .fp32) .log (m.binary (some .fp32) .real .add
        (m.literal (some .fp32) .real 1.0) (m.unary (some .fp32) .exp x)) else x
    else x
  let x := m.binary (some .fp32) .real .max x (m.literal (some .fp32) .real lower)
  let x := match upper with
    | .finite hi => if p.less x (m.literal (some .fp32) .real hi) then x
        else m.literal (some .fp32) .real hi
    | .posInf => m.unary (some .fp32) .minPosInf x
  if active then x else m.literal (some .fp32) .real 0.0

def dtRaw {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (l : Launch) (s : State α) (i j : Nat) : α :=
  if activeInput c.layout l i j then (s.mem r.dt (dtAddress c.layout l i j)).read .fp32
  else p.model.cast (some .fp32) .real .fp32 (p.model.literal (some .fp32) .real 0.0)

def biasRaw {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (l : Launch) (s : State α) (i : Nat) : α :=
  if head c.layout l i < c.heads then (s.mem r.bias (head c.layout l i*c.biasHead)).read .real
  else p.model.literal (some .fp32) .real 0.0

def preparedTile {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (s : State α) (i : TileIndex [c.BH,c.BC]) : α :=
  prepare c p hasBias useSoftplus lower upper (decide (activeInput c.layout l i.1.val i.2.1.val))
    (dtRaw c p r l s i.1.val i.2.1.val) (biasRaw c p r l s i.1.val)

def rateRaw {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (l : Launch) (mem : RegionName → Nat → Cell α) (i : Nat) : α :=
  p.model.cast (some .fp32) .fp32 .real
    (if head c.layout l i < c.heads then (mem r.rate (head c.layout l i*c.rateHead)).read .fp32
     else p.model.cast (some .fp32) .real .fp32 (p.model.literal (some .fp32) .real 0.0))

def writeCell {α : Type} (mem : RegionName → Nat → Cell α)
    (r : RegionName) (off : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun rg a => if rg = r ∧ a = off then v else mem rg a

def scatter {α : Type} (c : Whole.Host) (l : Launch) (r : RegionName)
    (value : TileIndex [c.BH,c.BC] → α) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [c.BH,c.BC]).foldl (fun m i =>
    if activeOutput c.layout l i.1.val i.2.1.val then
      writeCell m r (outAddress c.layout l i.1.val i.2.1.val) (.mk .real (value i)) else m) mem

def tileMemory {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (s : State α) : RegionName → Nat → Cell α :=
  let prepared := preparedTile c p r hasBias useSoftplus lower upper l s
  let middle := scatter c l r.out prepared s.mem
  let scaled := fun i : TileIndex [c.BH,c.BC] => p.model.binary (some .fp32) .real .mul
    (prepared i) (rateRaw c p r l middle i.1.val)
  let scanned := fun i : TileIndex [c.BH,c.BC] =>
    (p.plan.tree i.1 i.2.1).eval (p.model.binary (some .fp32) .real .add) p.fused (fun k => scaled (i.1,k,PUnit.unit))
  scatter c l r.sum scanned middle

theorem scatter_mem {α ι : Type} (r : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      lanes.foldl (fun m i => if active i then writeCell m r (off i) (value i) else m) s.mem := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi,if_true,State.write,writeCell] using ih (s.write r (off i) (value i))
    · simpa only [hi,if_false] using ih s


end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Backend
