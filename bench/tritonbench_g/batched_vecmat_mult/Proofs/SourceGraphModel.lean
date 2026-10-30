import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraph
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Float.TermModel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
abbrev Word := Expr
abbrev Memory := RegionName → Nat → Cell Word
abbrev Parameters := ValueProofs.Parameters

/-- A reduction may select its own valid source tree from the tagged raw
operands. This retains native shared-writer choice separately at every
program, output coordinate and K block; no equality between choices is used. -/
abbrev PlanSelector := ∀ n, (Fin n → Expr) → ReductionPlan n

def fixedPlans (plans : ∀ n, ReductionPlan n) : PlanSelector := fun n _ => plans n

/-- Complete selected Python body, including dimension and PID INT64 casts. -/
def source (p : Parameters) := VeriTile.Bench.Repairs.BatchedVecmatMult.source p

/-- Capture every source product, reduction seed and addition as syntax. -/
def treeValue (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => Expr.zero
  | .add a b => Expr.add (treeValue xs a) (treeValue xs b)

def sumValues (plans : PlanSelector)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => let args := fun i => xs (TileShape.insertAxisIndex shape ax out i)
      treeValue args (plans _ args).tree
  · exact fun out => let args := fun i => xs (TileShape.replaceAxisIndex shape ax out i)
      treeValue args (plans _ args).tree

/-- Execute the complete selected typed source in an expression-capturing algebra.
This algebra records operations; native word evaluation occurs only after the
explicit contraction pass in SourceGraph. -/
def engine (rest : Algebra Word) (plans : PlanSelector) : Algebra Word :=
  { rest with
    literal := fun pr d x => if x = 0 then Expr.zero else rest.literal pr d x
    binary := fun pr d op x y => match d, op with
      | .real, .add | .fp32, .add => Expr.add x y
      | .real, .mul | .fp32, .mul => Expr.mul x y
      | _, _ => rest.binary pr d op x y
    fp32Load := id
    reduceSum := fun _ => sumValues plans }

def row (p : Parameters) (s : State Word) (i : Nat) := s.pids 0 * p.BM + i
def col (p : Parameters) (s : State Word) (j : Nat) := s.pids 1 * p.BN + j
def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.BM,p.BN]) :=
  row p s i.1.val * p.N + col p s i.2.1.val
def aOffset (p : Parameters) (s : State Word) (i c k : Nat) :=
  row p s i * p.K + (c * p.BK + k)
def bOffset (p : Parameters) (s : State Word) (i j c k : Nat) :=
  row p s i * p.N * p.K + col p s j * p.K + (c * p.BK + k)

def blockSum (plans : PlanSelector) (p : Parameters)
    (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) : Word :=
  let args := fun k : Fin p.BK => Expr.mul
    ((s.mem p.A (aOffset p s i.1.val c k.val)).read .real)
    ((s.mem p.B (bOffset p s i.1.val i.2.1.val c k.val)).read .real)
  treeValue args (plans p.BK args).tree

/-- Source operation graph with the initial +0 and every carry addition. -/
def accumulated (plans : PlanSelector) (p : Parameters)
    (s : State Word) : Nat → TileIndex [p.BM,p.BN] → Word
  | 0, _ => Expr.zero
  | c + 1, i => Expr.add (accumulated plans p s c i) (blockSum plans p s c i)

def expectedMemory (plans : PlanSelector)
    (p : Parameters) (s : State Word) : Memory :=
  (TileShape.allIndices [p.BM,p.BN]).foldl (fun mem i =>
    fun r o => if r = p.output ∧ o = outputOffset p s i then
      .mk .real (accumulated plans p s (p.K / p.BK) i) else mem r o) s.mem

def Outside (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.BM,p.BN], o ≠ outputOffset p s i

structure Invariant (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (c : Nat) (current : State Word) : Prop where
  dimn : current.regs .int [] "dim_n" = some (fun _ => (p.N : Int))
  dimk : current.regs .int [] "dim_k" = some (fun _ => (p.K : Int))
  mi : current.regs .int [] "m_index" = some (fun _ => (initial.pids 0 : Int))
  ni : current.regs .int [] "n_index" = some (fun _ => (initial.pids 1 : Int))
  offsets : current.regs .int [p.BM,p.BN] "output_tile" = some (fun i => (outputOffset p initial i : Int))
  carry : current.regs .real [p.BM,p.BN] "vecmat" = some (accumulated plans p initial c)
  blocks : current.regs .int [] "k_blocks" = some (fun _ => ((p.K / p.BK : Nat) : Int))
  memory : current.mem = initial.mem


theorem source_admitted (p : Parameters) :
    (source p).toAlgorithm? = .ok (source p).toAlgKernel := by rfl

theorem expectedMemory_frame (plans : PlanSelector)
    (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    expectedMemory plans p s r o = s.mem r o := by
  have aux : ∀ (xs : List (TileIndex [p.BM,p.BN])) (acc : Memory),
      xs.foldl (fun mem i => fun r o => if r = p.output ∧ o = outputOffset p s i then
        .mk .real (accumulated plans p s (p.K / p.BK) i) else mem r o) acc r o = acc r o := by
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
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
