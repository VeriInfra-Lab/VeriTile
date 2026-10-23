import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawSource
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program Backend
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Primitives (α : Type) (c : Whole.Host) extends Backend.Primitives α c where
  minimum : α → α → α

def sourceRun {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) :
    {v : State α // RawSource.exec p.model p.minimum c (RawSource.source c r hasBias useSoftplus)
      lower upper l s = some v} := by
  cases hasBias <;> cases useSoftplus
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 900000) [RawSource.exec,RawSource.source,Source.prefixBeforeStore,WholeFP.typed32,Program.kernel,source,finiteSource,infiniteSource,
      ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,natLt,natLe,numericLt,numericLe,WholeFP.tileState,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,Whole.Host.layout,scatter_regs,p.lt_support,p.le_support,p.scan_support]
    try rfl

/-- Independent scalar preparation graph. Storage/load casts, both boolean
branches, cutoff softplus, finite/default clamp and the final remask remain. -/
def prepare {α : Type} (c : Whole.Host) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α)
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
  let x := m.binary (some .fp32) .real .max x lower
  let x := p.minimum x upper
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
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) (i : TileIndex [c.BH,c.BC]) : α :=
  prepare c p hasBias useSoftplus lower upper (decide (activeInput c.layout l i.1.val i.2.1.val))
    (dtRaw c p r l s i.1.val i.2.1.val) (biasRaw c p r l s i.1.val)

def rateRaw {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (l : Launch) (mem : RegionName → Nat → Cell α) (i : Nat) : α :=
  p.model.cast (some .fp32) .fp32 .real
    (if head c.layout l i < c.heads then (mem r.rate (head c.layout l i*c.rateHead)).read .fp32
     else p.model.cast (some .fp32) .real .fp32 (p.model.literal (some .fp32) .real 0.0))

def tileMemory {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) : RegionName → Nat → Cell α :=
  let prepared := preparedTile c p r hasBias useSoftplus lower upper l s
  let middle := scatter c l r.out prepared s.mem
  let scaled := fun i : TileIndex [c.BH,c.BC] => p.model.binary (some .fp32) .real .mul
    (prepared i) (rateRaw c p r l middle i.1.val)
  let scanned := fun i : TileIndex [c.BH,c.BC] =>
    (p.plan.tree i.1 i.2.1).eval (p.model.binary (some .fp32) .real .add) p.fused (fun k => scaled (i.1,k,PUnit.unit))
  scatter c l r.sum scanned middle

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
