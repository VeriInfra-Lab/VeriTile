import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBackend
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
abbrev Row (p : Parameters) := TileIndex [p.block]
abbrev Lane (p : Parameters) := TileIndex [p.block,p.blockTokens]
def row {α : Type} (p : Parameters) (s : State α) (i : Row p) := s.pids 0*p.block+i.1.val
def active {α : Type} (p : Parameters) (s : State α) (i : Row p) : Prop := row p s i<p.batches
instance {p : Parameters} {s : State α} {i : Row p} : Decidable (active p s i) := inferInstanceAs (Decidable (_<_))
def intLoad (s : State α) (r : RegionName) (o : Nat) (enabled : Bool) : Int :=
  if enabled then (s.mem r o).read .int else s.undef .int r o

def seedValue (p : Parameters) (s : State α) (i : Row p) : Int :=
  intLoad s p.seeds (row p s i) (decide (active p s i))
def offsetValue (p : Parameters) (s : State α) (i : Row p) : Int :=
  (BitVec.ofInt 32 (intLoad s p.offsets (row p s i) (decide (active p s i)))).toInt
/-- I64 storage typing includes masked load results: every native undefined
seed is still an arbitrary signed64 payload. No seed or offset is fixed. -/
def SeedDomain (p : Parameters) (s : State α) : Prop :=
  ∀ i : Row p, Random.IntegerType.i64.inRange (seedValue p s i) = Bool.true

def sample (P : Primitives α) (p : Parameters) (s : State α) (i : Row p) : α :=
  P.bits32 ⟨(Random.rand (seedValue p s i) (offsetValue p s i) .bits32 10).bits⟩

def initialOutput (p : Parameters) (s : State α) (i : Row p) : Int :=
  intLoad s p.indices (row p s i*p.strideIndexBatch) (decide (active p s i))

def scoreOffset (p : Parameters) (s : State α) (b : Nat) (i : Lane p) :=
  row p s (i.1,PUnit.unit)*p.strideScoreBatch+(b+i.2.1.val)*p.strideScoreToken

def padding (P : Primitives α) : Storage → α
  | .fp32 => P.zero (some .fp32) .real
  | .fp64 => P.zero (some .fp64) .real
  | .fp16 => P.cast (some .fp32) .real .fp16 (P.zero (some .fp32) .real)
  | .bf16 => P.cast (some .fp32) .real .bf16 (P.zero (some .fp32) .real)

def readScore (d : Storage) (c : Cell α) : α :=
  match d with
  | .fp32 | .fp64 => c.read .real | .fp16 => c.read .fp16 | .bf16 => c.read .bf16

def scores (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (i : Lane p) : α :=
  P.cast (some .fp32) d.dtype .real
    (if active p s (i.1,PUnit.unit) ∧ b+i.2.1.val<p.tokens then
      readScore d (s.mem p.scores (scoreOffset p s b i)) else padding P d)

def scanned (P : Primitives α) (d : Storage) (p : Parameters) (s : State α) (b : Nat) : Lane p → α :=
  P.scan none .sum ⟨1,by simp⟩ .forward (scores P d p s b)

structure Accumulator (α : Type) (p : Parameters) where
  acc : Row p → α
  output : Row p → Int

def cumulative (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) (i : Lane p) : α :=
  P.add (st.acc (i.1,PUnit.unit)) (scanned P d p s b i)
def valid (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) (i : Lane p) : Bool :=
  P.lt (P.sub (cumulative P d p s b st i) (scores P d p s b i)) (sample P p s (i.1,PUnit.unit)) &&
    P.le (sample P p s (i.1,PUnit.unit)) (cumulative P d p s b st i)
def found (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) (i : Row p) : Bool :=
  decide (0<TileShape.discreteBoolSum ⟨1,by simp⟩ Bool.false (valid P d p s b st) i)
def selected (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) (i : Row p) : Nat :=
  b+TileShape.discreteIntArgMax ⟨1,by simp⟩ (fun j => if valid P d p s b st j then (1:Int) else 0) i

def selectedIndex (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) (i : Row p) : Int :=
  if found P d p s b st i && decide (active p s i) then
    (s.mem p.indices (row p s i*p.strideIndexBatch+min (selected P d p s b st i) (p.tokens-1)*p.strideIndexToken)).read .int
  else -1

/-- Independent source algorithm on initial memory. A later matching block
replaces the previous index, even when rounded intervals overlap. -/
def advance (P : Primitives α) (d : Storage) (p : Parameters) (s : State α)
    (b : Nat) (st : Accumulator α p) : Accumulator α p :=
  { acc := fun i => P.add (st.acc i) (P.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned P d p s b) i)
    output := fun i => if found P d p s b st i then selectedIndex P d p s b st i else st.output i }

def initialAccumulator (P : Primitives α) (p : Parameters) (s : State α) : Accumulator α p :=
  ⟨fun _ => P.zero (some .fp32) .real,initialOutput p s⟩
def rounds (P : Primitives α) (d : Storage) (p : Parameters) (s : State α) : Nat → Accumulator α p
  | 0 => initialAccumulator P p s
  | k+1 => advance P d p s (k*p.blockTokens) (rounds P d p s k)
def count (p : Parameters) := (p.tokens+p.blockTokens-1)/p.blockTokens

def written (P : Primitives α) (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i =>
    if active p s i then t.write p.output (row p s i) (.mk .int ((rounds P d p s (count p)).output i)) else t) s

#specNonCircular written avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
