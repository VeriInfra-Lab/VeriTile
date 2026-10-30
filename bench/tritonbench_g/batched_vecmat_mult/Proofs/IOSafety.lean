import bench.tritonbench_g.batched_vecmat_mult.IO
import bench.tritonbench_g.batched_vecmat_mult.Proofs.KernelSafety
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphAccess

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.IOSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open VecmatIO
open SourceGraph (Expr)
noncomputable section

def Windows (p : Parameters) (bounds : RegionName → Nat) : Prop :=
  (∀ (m : Fin p.M) (k : Fin p.K), m.val*p.K+k.val < bounds p.A) ∧
  (∀ (m : Fin p.M) (n : Fin p.N) (k : Fin p.K),
    m.val*p.N*p.K+n.val*p.K+k.val < bounds p.B) ∧
  (∀ (m : Fin p.M) (n : Fin p.N), m.val*p.N+n.val < bounds p.output)

theorem launch_safe (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (bounds : RegionName → Nat) (windows : Windows p bounds)
    (pid : Nat) (valid : pid<SourceGraph.gridCount p) (s : State Expr) :
    (launch rest profile p pid).Safe bounds s := by
  obtain ⟨hm,hn⟩ := SourceGraph.coordinate_bounds p pid valid
  have row (i : Fin p.BM) := SourceGraph.tiled_coordinate_lt p.M p.BM ⟨_,hm⟩ i
  have col (j : Fin p.BN) := SourceGraph.tiled_coordinate_lt p.N p.BN ⟨_,hn⟩ j
  apply KernelSafety.source_safe rest (KernelProfile.toPlans profile) p bounds
  · intro c hc i k
    exact windows.1 ⟨_,row i⟩ ⟨_,SourceGraph.tiled_coordinate_lt p.K p.BK ⟨c,hc⟩ k⟩
  · intro c hc i j k
    exact windows.2.1 ⟨_,row i⟩ ⟨_,col j⟩ ⟨_,SourceGraph.tiled_coordinate_lt p.K p.BK ⟨c,hc⟩ k⟩
  · intro i
    exact windows.2.2 ⟨_,row i.1⟩ ⟨_,col i.2.1⟩

theorem jobs (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (bounds : RegionName → Nat) (windows : Windows p bounds)
    (jobs : List Nat) (valid : ∀ pid∈jobs,pid<SourceGraph.gridCount p) (s : State Expr) :
    ScheduleSafe bounds (jobs.map (launch rest profile p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    exact ⟨launch_safe rest profile p bounds windows pid (valid pid (by simp)) s,
      fun t _ => ih (fun q hq => valid q (by simp [hq])) t⟩

theorem grid (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (bounds : RegionName → Nat) (windows : Windows p bounds) (s : State Expr) :
    ScheduleSafe bounds (schedule rest profile p) s :=
  jobs rest profile p bounds windows _ (fun _ h => List.mem_range.mp h) s

#axiomsClean grid
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.IOSafety
