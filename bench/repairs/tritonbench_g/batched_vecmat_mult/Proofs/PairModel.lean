import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphTensor

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr Word Memory Parameters PlanSelector engine treeValue sumValues
  row col outputOffset aOffset bOffset blockSum Outside)
noncomputable section

def source := VeriTile.Bench.Repairs.BatchedVecmatMult.blockFirst

/-- Source operation graph with the initial +0 and every carry addition. -/
def accumulated (plans : PlanSelector) (p : Parameters)
    (s : State Word) : Nat → TileIndex [p.BM,p.BN] → Word
  | 0, _ => Expr.zero
  | c + 1, i => Expr.add (blockSum plans p s c i) (accumulated plans p s c i)

def expectedMemory (plans : PlanSelector)
    (p : Parameters) (s : State Word) : Memory :=
  (TileShape.allIndices [p.BM,p.BN]).foldl (fun mem i =>
    fun r o => if r = p.output ∧ o = outputOffset p s i then
      .mk .real (accumulated plans p s (p.K / p.BK) i) else mem r o) s.mem

def Outside (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.BM,p.BN], o ≠ outputOffset p s i

structure Invariant (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (c : Nat) (current : State Word) : Prop where
  mi : current.regs .nat [] "m_index" = some (fun _ => initial.pids 0)
  ni : current.regs .nat [] "n_index" = some (fun _ => initial.pids 1)
  offsets : current.regs .nat [p.BM,p.BN] "output_tile" = some (outputOffset p initial)
  carry : current.regs .real [p.BM,p.BN] "vecmat" = some (accumulated plans p initial c)
  blocks : current.regs .nat [] "k_blocks" = some (fun _ => p.K / p.BK)
  memory : current.mem = initial.mem

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


end
end VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
