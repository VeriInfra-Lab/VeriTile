import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Parameters where
  (a b c : RegionName)
  (rows columns inner aRow aInner bInner bColumn cRow cColumn
    blockRows blockColumns blockInner : Nat)

/-- Both constexpr output branches of the original one-tile kernel. The explicit
load dtypes recover the block-pointer element types. The output annotation
materializes the storage conversion, including the fp32 same-format case.
No boundary checks, masks, zero padding, or additional tiles are introduced. -/
def typed (dtype out_dtype : FloatDType) (p : Parameters) : ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=$(p.a), shape=($(p.rows), $(p.inner)),
    strides=($(p.aRow), $(p.aInner)), offsets=($(0), $(0)),
    block_shape=($(p.blockRows), $(p.blockInner)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=$(p.b), shape=($(p.inner), $(p.columns)),
    strides=($(p.bInner), $(p.bColumn)), offsets=($(0), $(0)),
    block_shape=($(p.blockInner), $(p.blockColumns)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=$(p.c), shape=($(p.rows), $(p.columns)),
    strides=($(p.cRow), $(p.cColumn)), offsets=($(0), $(0)),
    block_shape=($(p.blockRows), $(p.blockColumns)), order=(1, 0))
  a = tl.load(a_block_ptr, dtype=DTYPE)
  b = tl.load(b_block_ptr, dtype=DTYPE)
  c = (tl.dot(a, b)).to(OUT_DTYPE)
  tl.store(c_block_ptr, c)
}

def outputDType (half : Bool) : FloatDType := if half then .fp16 else .fp32

def source (p : Parameters) (half : Bool) : ComputeKernel :=
  typed .fp16 (outputDType half) p

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

/-- Independent initial-memory value, for every tile cell and both output
formats. The dot sees the original inputs even if an output region aliases one
of them, because both complete source tiles are loaded before the first store. -/
def outputValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (i : CIndex p) : α :=
  M.cast none .fp32 (outputDType half)
    (D.apply (some .fp32) {} (batch := []) (aValues p s) (bValues p s) i)

end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
