import bench.tritonbench_g.batched_vecmat_mult.Kernels
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Float.TermModel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev Memory := RegionName → Nat → Cell Word


def source (p : Parameters) := TypedSource.original p.A p.B p.output p.M p.N p.K p.BM p.BN p.BK

/-- Every product and explicit seed is retained. The plan asserts leaf
multiplicity only, with no real-sum or reassociation identity. -/
def treeValue (cfg : FP.Config) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a b => FP.add cfg (treeValue cfg xs a) (treeValue cfg xs b)

def sumValues (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => treeValue cfg (fun i => xs (TileShape.insertAxisIndex shape ax out i)) (plans _).tree
  · exact fun out => treeValue cfg (fun i => xs (TileShape.replaceAxisIndex shape ax out i)) (plans _).tree

/-- The existing float32 DSL source uses the historical real channel, whose
carrier here is concrete fp32 words, never mathematical real arithmetic. -/
def engine (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Algebra Word :=
  { rest with
    literal := fun pr d x => if x = 0 then FP.Value.zero .fp32 else rest.literal pr d x
    binary := fun pr d op x y => match d, op with
      | .real, .add | .fp32, .add => FP.add cfg x y
      | .real, .mul | .fp32, .mul => FP.mul cfg x y
      | _, _ => rest.binary pr d op x y
    fp32Load := id
    reduceSum := fun _ => sumValues cfg plans }

def row (p : Parameters) (s : State Word) (i : Nat) := s.pids 0 * p.BM + i
def col (p : Parameters) (s : State Word) (j : Nat) := s.pids 1 * p.BN + j
def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.BM,p.BN]) :=
  row p s i.1.val * p.N + col p s i.2.1.val
def aOffset (p : Parameters) (s : State Word) (i c k : Nat) :=
  row p s i * p.K + (c * p.BK + k)
def bOffset (p : Parameters) (s : State Word) (i j c k : Nat) :=
  row p s i * p.N * p.K + col p s j * p.K + (c * p.BK + k)

def blockSum (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (p : Parameters)
    (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) : Word :=
  treeValue cfg (fun k : Fin p.BK => FP.mul cfg
    ((s.mem p.A (aOffset p s i.1.val c k.val)).read .real)
    ((s.mem p.B (bOffset p s i.1.val i.2.1.val c k.val)).read .real)) (plans p.BK).tree

/-- Ordered block accumulation, retaining the initial fp32 +0 and every
rounded add. No commutativity, zero identity or exact-real sum is assumed. -/
def accumulated (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (p : Parameters)
    (s : State Word) : Nat → TileIndex [p.BM,p.BN] → Word
  | 0, _ => FP.Value.zero .fp32
  | c + 1, i => FP.add cfg (accumulated cfg plans p s c i) (blockSum cfg plans p s c i)

def expectedMemory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) : Memory :=
  (TileShape.allIndices [p.BM,p.BN]).foldl (fun mem i =>
    fun r o => if r = p.output ∧ o = outputOffset p s i then
      .mk .real (accumulated cfg plans p s (p.K / p.BK) i) else mem r o) s.mem

def Outside (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.BM,p.BN], o ≠ outputOffset p s i

structure Invariant (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (c : Nat) (current : State Word) : Prop where
  mi : current.regs .nat [] "m_index" = some (fun _ => initial.pids 0)
  ni : current.regs .nat [] "n_index" = some (fun _ => initial.pids 1)
  offsets : current.regs .nat [p.BM,p.BN] "output_tile" = some (outputOffset p initial)
  carry : current.regs .real [p.BM,p.BN] "vecmat" = some (accumulated cfg plans p initial c)
  blocks : current.regs .nat [] "k_blocks" = some (fun _ => p.K / p.BK)
  memory : current.mem = initial.mem

theorem source_admitted (p : Parameters) :
    (source p).toAlgorithm? = .ok (source p).toAlgKernel := by rfl

theorem expectedMemory_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    expectedMemory cfg plans p s r o = s.mem r o := by
  have aux : ∀ (xs : List (TileIndex [p.BM,p.BN])) (acc : Memory),
      xs.foldl (fun mem i => fun r o => if r = p.output ∧ o = outputOffset p s i then
        .mk .real (accumulated cfg plans p s (p.K / p.BK) i) else mem r o) acc r o = acc r o := by
    intro xs
    induction xs with
    | nil => intro acc; rfl
    | cons i xs ih =>
      intro acc
      rw [List.foldl_cons, ih]
      have hn : ¬ (r = p.output ∧ o = outputOffset p s i) := by
        intro he
        rcases h with hr | ho
        · exact hr he.1
        · exact ho i he.2
      exact if_neg hn
  exact aux _ _

#specNonCircular expectedMemory avoiding [source, FP.Structural.exec, FP.Structural.run]
#axiomsClean source_admitted
#axiomsClean expectedMemory_frame
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
