import bench.tritonbench_g.layer_norm_ops.Proofs.FPTypedSource
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
noncomputable section

def op {α : Type} (M : Algebra α) (b : Binary) := M.binary (some .fp32) .real b
def zero {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 0
def pad {α : Type} (M : Algebra α) :=
  M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0.0)
def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (base : Nat) (i : TileIndex [p.block]) : α :=
  M.cast (some .fp32) .fp32 .real
    (if i.1.val<p.columns then (s.mem r (base+i.1.val)).read .fp32 else pad M)
def weight {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) : α := M.cast (some .fp32) .fp32 .real
  (if i.1.val<p.columns then (s.mem p.weight i.1.val).read .fp32 else s.undef .fp32 p.weight i.1.val)
def scalar {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (r : RegionName) :=
  M.cast (some .fp32) .fp32 .real ((s.mem r (s.pids 0)).read .fp32)
def dyValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  loaded M p s p.dy (s.pids 0*p.dyStride)
def xhatValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) (i : TileIndex [p.block]) :=
  if i.1.val<p.columns then
    op M .mul (if plain then op M .sub (loaded M p s p.x (s.pids 0*p.xStride) i)
      (scalar M s p.mean) else loaded M p s p.x (s.pids 0*p.xStride) i) (scalar M s p.rstd)
  else M.literal (some .fp32) .real 0.0
def wdyValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) := op M .mul (weight M p s i) (dyValue M p s i)
def average {α : Type} (M : Algebra α) (p : Parameters) (values : TileIndex [p.block] → α) :=
  op M .div (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false values PUnit.unit)
    (M.literal (some .fp32) .real (p.columns : ℝ))
def rowC1 {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (plain : Bool) :=
  average M p (fun i => op M .mul (xhatValue M p s plain i) (wdyValue M p s i))
def rowC2 {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  average M p (wdyValue M p s)
def rowDX {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) (i : TileIndex [p.block]) :=
  op M .mul (op M .sub (wdyValue M p s i)
    (if plain then op M .add (op M .mul (xhatValue M p s plain i) (rowC1 M p s plain)) (rowC2 M p s)
      else op M .mul (xhatValue M p s plain i) (rowC1 M p s plain))) (scalar M s p.rstd)
/-- The native one-row parameter-gradient accumulator still performs +0+term.
This cannot be simplified to term in the full signed-zero/nonfinite domain. -/
def rowDW {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) (i : TileIndex [p.block]) :=
  op M .add (zero M) (op M .mul (dyValue M p s i) (xhatValue M p s plain i))
def rowDB {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) := op M .add (zero M) (dyValue M p s i)
def suppliedXhat {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  loaded M p s p.xhat (s.pids 0*p.xhatStride)
def c1Value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  average M p (fun i => op M .mul (suppliedXhat M p s i) (wdyValue M p s i))
def suppliedDX {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) (i : TileIndex [p.block]) :=
  op M .mul (op M .sub (wdyValue M p s i)
    (if plain then op M .add (op M .mul (suppliedXhat M p s i) (scalar M s p.c1)) (scalar M s p.c2)
      else op M .mul (suppliedXhat M p s i) (scalar M s p.c1))) (scalar M s p.rstd)
def residualValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.block]) := op M .add
  (loaded M p s p.dxBase (s.pids 0*p.dxStride) i) (loaded M p s p.dres (s.pids 0*p.dresStride) i)
def stored {α : Type} (M : Algebra α) (v : α) : Cell α := .mk .fp32 (M.cast none .real .fp32 v)
def offset {α : Type} (s : State α) (stride : Nat) (i : TileIndex [B]) := s.pids 0*stride+i.1.val

def scatter {α : Type} (p : Parameters) (s : State α) (r : RegionName) (stride : Nat)
    (values : TileIndex [p.block] → Cell α) (t : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val<p.columns then
    t.write r (offset s stride i) (values i) else t) t

def rowReference {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) : State α :=
  let t := scatter p s p.dx p.dxStride (fun i => stored M (rowDX M p s plain i)) s
  let u := scatter p s p.dw p.columns (fun i => stored M (rowDW M p s plain i)) t
  if plain then scatter p s p.db p.columns (fun i => stored M (rowDB M p s i)) u else u

def dxReference {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) := scatter p s p.dx p.dxStride (fun i => stored M (suppliedDX M p s plain i)) s

def residualReference {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  let t := scatter p s p.dresIn p.dresInStride (fun i => stored M (residualValue M p s i)) s
  scatter p s p.dx p.dxStride (fun i => stored M (residualValue M p s i)) t

def OutsideVector {α : Type} (p : Parameters) (s : State α) (r : RegionName) (stride : Nat)
    (r' : RegionName) (o : Nat) := r' ≠ r ∨ ∀ i : TileIndex [p.block],
      i.1.val<p.columns → o ≠ offset s stride i
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
