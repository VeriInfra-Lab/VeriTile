import bench.tritonbench_g.matmul_tma.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows, p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner, p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]

def aOffset (p : Parameters) (i : AIndex p) := i.1.val * p.aRow + i.2.1.val * p.aInner
def bOffset (p : Parameters) (i : BIndex p) := i.1.val * p.bInner + i.2.1.val * p.bColumn
def cOffset (p : Parameters) (i : CIndex p) := i.1.val * p.cRow + i.2.1.val * p.cColumn

/-- Dot is an explicit primitive, retaining the compiler precision and dot
options. Identifying it with a native tensor instruction is a separate binding.
The structural theorem assumes no associativity or real matrix-product law. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : Dot α) : Algebra α :=
  {rest with dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
    match left, right, out with
    | .fp16, .fp16, .fp32 => some (D.apply precision options (batch := batch) (m := m) (k := k) (n := n) a b)
    | left, right, out => rest.dotTyped precision left right out options a b}

def aValues {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : AIndex p) : α :=
  (s.mem p.a (aOffset p i)).read .fp16

def bValues {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : BIndex p) : α :=
  (s.mem p.b (bOffset p i)).read .fp16

/-- Source-ordered explicit FP16 conversion (when selected), followed by the
implicit typed store conversion. Both retain their actual precision context. -/
def outputValue (M : Algebra α) (half : Bool) (value : α) : α :=
  if half then M.cast none .fp16 .fp16 (M.cast none .fp32 .fp16 value)
  else M.cast none .fp32 .fp32 value

/-- Independent initial-memory value, for every tile cell and both output
formats. The dot sees the original inputs even if an output region aliases one
of them, because both complete source tiles are loaded before the first store. -/
def outputValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (i : CIndex p) : α :=
  outputValue M half
    (D.apply none {} (batch := []) (aValues p s) (bValues p s) i)

end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
