import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardModel
import bench.tritonbench_g.parallel_attention.Proofs.FPSources

namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

def prelude (p : BackwardParameters) := (backwardKernel p).surfaceBody.take 15
def body (p : BackwardParameters) : List ComputeStmt :=
  let code : List ComputeStmt := (backwardKernel p).surfaceBody
  match code[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def postlude (p : BackwardParameters) := (backwardKernel p).surfaceBody.drop 16

theorem source_split (p : BackwardParameters) : (backwardKernel p).surfaceBody = prelude p ++
    [.forRangeDyn "row" (Op.ref .nat [] "row_start") (Op.ref .nat [] "row_end")
      (Op.constNat 1) (body p)] ++ postlude p := by rfl

abbrev binary := @Forward.binary
abbrev unhintedBinary := @Forward.unhintedBinary
abbrev literal := @Forward.literal
abbrev padding := @Forward.padding
abbrev cast := @Forward.cast
abbrev stored := @Forward.stored

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (s : State α) (region : RegionName) (base : Nat) (i : TileIndex [p.block]) : α :=
  cast M (if i.1.val < p.columns then (s.mem region (base+i.1.val)).read .fp32 else padding M)

structure Acc (α : Type) (p : BackwardParameters) where
  state : State α
  (weight bias dw db : TileIndex [p.block] → α)

def initial {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters) (s : State α) : Acc α p :=
  ⟨s, loaded M p s p.weight 0, loaded M p s p.bias 0,
    fun _ => literal M 0, fun _ => literal M 0⟩

def meanValue {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) := M.cast none .fp32 .real ((a.state.mem p.mean (row*p.meanStride)).read .fp32)
def rstdValue {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) := cast M ((a.state.mem p.rstd (row*p.rstdStride)).read .fp32)

def xhat {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) :=
  if i.1.val < p.columns then
    let x := loaded M p a.state p.x (row*p.xStride) i
    unhintedBinary M .mul (if p.rms then x else unhintedBinary M .sub x (meanValue M p a row))
      (M.cast none .fp32 .real ((a.state.mem p.rstd (row*p.rstdStride)).read .fp32))
  else M.literal none .real 0

def dy {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) := loaded M p a.state p.dy (row*p.dyStride)
def wdy {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) := binary M .mul (a.weight i) (dy M p a row i)
def average {α : Type} (M : Algebra α) (p : BackwardParameters) (f : TileIndex [p.block] → α) :=
  binary M .div (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false f PUnit.unit) (literal M (p.columns : ℝ))
def c1 {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters) (a : Acc α p) (row : Nat) :=
  average M p (fun i => binary M .mul (xhat M p a row i) (wdy M p a row i))
def c2 {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters) (a : Acc α p) (row : Nat) :=
  average M p (wdy M p a row)
def dxBase {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) :=
  binary M .mul (binary M .sub (wdy M p a row i)
    (let projection := binary M .mul (xhat M p a row i) (c1 M p a row)
     if p.rms then projection else binary M .add projection (c2 M p a row))) (rstdValue M p a row)
def yValue {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) :=
  let result := unhintedBinary M .mul (xhat M p a row i) (a.weight i)
  if p.hasBias then unhintedBinary M .add result (a.bias i) else result

def scatter {α : Type} (p : BackwardParameters) (row : Nat) (region : RegionName) (stride : Nat)
    (values : TileIndex [p.block] → Cell α) (s : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val < p.columns then
    t.write region (row*stride+i.1.val) (values i) else t) s

def recomputed {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) :=
  if p.recompute then scatter p row p.y p.yStride (fun i => stored M (yValue M p a row i)) a.state
  else a.state

def dxValue {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) :=
  let dx := dxBase M p a row i
  if p.hasDres then binary M .add dx (loaded M p (recomputed M p a row) p.dres (row*p.dresStride) i)
  else dx

def update {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (row : Nat) (a : Acc α p) : Acc α p where
  state :=
    let t := recomputed M p a row
    let u := if p.storeDres then scatter p row p.dresIn p.dresInStride
      (fun i => stored M (dxValue M p a row i)) t else t
    scatter p row p.dx p.dxStride (fun i => stored M (dxValue M p a row i)) u
  weight := a.weight
  bias := a.bias
  dw := fun i => binary M .add (a.dw i) (binary M .mul (dy M p a row i) (xhat M p a row i))
  db := if p.hasBias then fun i => binary M .add (a.db i) (dy M p a row i) else a.db

def start {α : Type} (p : BackwardParameters) (s : State α) := s.pids 0*p.rowsPerProgram
def stop {α : Type} (p : BackwardParameters) (s : State α) := min ((s.pids 0+1)*p.rowsPerProgram) p.rows
def accumulated {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters) (s : State α) :=
  foldRange (update M p) (start p s) (stop p s) 1 (initial M p s)
def finish {α : Type} (M : Algebra α) (p : BackwardParameters) (pid : Nat) (a : Acc α p) :=
  let t := scatter p pid p.dw p.columns (fun i => stored M (a.dw i)) a.state
  if p.hasBias then scatter p pid p.db p.columns (fun i => stored M (a.db i)) t else t
def reference {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters) (s : State α) :=
  finish M p (s.pids 0) (accumulated M p s)

#axiomsClean source_split
end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
