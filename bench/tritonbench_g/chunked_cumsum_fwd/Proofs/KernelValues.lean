import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton FP.Structural
open Math
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def writeCell {α : Type} (mem : RegionName → Nat → Cell α)
    (r : RegionName) (off : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun rg a => if rg = r ∧ a = off then v else mem rg a

def scatter {α : Type} (c : Geometry) (l : Launch) (r : RegionName)
    (value : TileIndex [c.BH,c.BC] → α) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [c.BH,c.BC]).foldl (fun m i =>
    if activeOutput c.layout l i.1.val i.2.1.val then
      writeCell m r (outAddress c.layout l i.1.val i.2.1.val) (.mk .real (value i)) else m) mem

/-- Independent scalar preparation graph. Storage/load casts, both boolean
branches, cutoff softplus, finite/default clamp and the final remask remain. -/
def prepare {α : Type} (c : Geometry) (p : Primitives α c)
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

def dtRaw {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (l : Launch) (s : State α) (i j : Nat) : α :=
  if activeInput c.layout l i j then (s.mem "dt" (dtAddress c.layout l i j)).read .fp32
  else p.model.cast (some .fp32) .real .fp32 (p.model.literal (some .fp32) .real 0.0)

def biasRaw {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (l : Launch) (s : State α) (i : Nat) : α :=
  if head c.layout l i < c.heads then (s.mem "bias" (head c.layout l i*c.biasHead)).read .real
  else p.model.literal (some .fp32) .real 0.0

def preparedTile {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) (i : TileIndex [c.BH,c.BC]) : α :=
  prepare c p hasBias useSoftplus lower upper (decide (activeInput c.layout l i.1.val i.2.1.val))
    (dtRaw c p l s i.1.val i.2.1.val) (biasRaw c p l s i.1.val)

def rateRaw {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (l : Launch) (mem : RegionName → Nat → Cell α) (i : Nat) : α :=
  p.model.cast (some .fp32) .fp32 .real
    (if head c.layout l i < c.heads then (mem "rate" (head c.layout l i*c.rateHead)).read .fp32
     else p.model.cast (some .fp32) .real .fp32 (p.model.literal (some .fp32) .real 0.0))

def tileMemory {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) : RegionName → Nat → Cell α :=
  let prepared := preparedTile c p hasBias useSoftplus lower upper l s
  let middle := scatter c l "out" prepared s.mem
  let scaled := fun i : TileIndex [c.BH,c.BC] => p.model.binary (some .fp32) .real .mul
    (prepared i) (rateRaw c p l middle i.1.val)
  let scanned := fun i : TileIndex [c.BH,c.BC] =>
    (p.plan.tree i.1 i.2.1).eval (p.model.binary (some .fp32) .real .add) p.fused (fun k => scaled (i.1,k,PUnit.unit))
  scatter c l "sum" scanned middle

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
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
