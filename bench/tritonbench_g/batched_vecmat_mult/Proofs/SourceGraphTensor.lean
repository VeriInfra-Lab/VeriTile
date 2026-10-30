import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphSchedule

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨Expr.zero⟩
set_option maxHeartbeats 2000000

/-- Tensor-coordinate reference, independent of program IDs, output memory,
source execution and the initial output allocation. -/
def tensorBlock (plans : PlanSelector)
    (p : Parameters) (mem : Memory) (m n c : Nat) : Word :=
  let args := fun e : Fin p.BK => Expr.mul ((mem p.A (m*p.K + (c*p.BK+e.val))).read .real)
    ((mem p.B (m*p.N*p.K+n*p.K+(c*p.BK+e.val))).read .real)
  treeValue args (plans p.BK args).tree

def tensorValue (plans : PlanSelector)
    (p : Parameters) (mem : Memory) (m n : Nat) : Nat → Word
  | 0 => Expr.zero
  | c+1 => Expr.add (tensorValue plans p mem m n c) (tensorBlock plans p mem m n c)

theorem accumulated_tensor (plans : PlanSelector)
    (p : Parameters) (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    accumulated plans p s c i =
      tensorValue plans p s.mem (row p s i.1.val) (col p s i.2.1.val) c := by
  induction c with
  | zero => rfl
  | succ c ih => rw [accumulated, tensorValue, ih]; rfl

/-- A complete schedule's reference actually stores each valid program lane.
The proof uses unique writers, not floating-point commutation. -/
theorem scheduled_readback (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (pid : Nat) (hp : pid < gridCount p)
    (i : TileIndex [p.BM,p.BN]) (jobs : List Nat) (mem : Memory)
    (valid : ∀ k ∈ jobs, k < gridCount p) (present : pid ∈ jobs) :
    scheduledInitialMemory plans p initial jobs mem p.output
      (outputOffset p (programState p initial initial.mem pid) i) =
      Cell.mk .real (accumulated plans p (programState p initial initial.mem pid)
        (p.K / p.BK) i) := by
  classical
  induction jobs generalizing mem with
  | nil => simp at present
  | cons k jobs ih =>
    change scheduledInitialMemory plans p initial jobs
      (applyInitialProgram plans p initial k mem) p.output _ = _
    by_cases tail : pid ∈ jobs
    · exact ih _ (fun q hq => valid q (List.mem_cons_of_mem _ hq)) tail
    · have head : pid = k := (List.mem_cons.mp present).resolve_right tail
      subst k
      rw [scheduled_memory_frame]
      · rw [applyInitialProgram, if_pos ⟨rfl,i,rfl⟩]
        exact actual_program_expected_readback plans p initial initial.mem pid hp i
      · intro k hk write
        obtain ⟨_, j, address⟩ := write
        have same := (actual_program_output_unique p initial initial.mem initial.mem pid k hp
          (valid k (List.mem_cons_of_mem _ hk)) i j address).1
        exact tail (same ▸ hk)

/-- Every tensor coordinate is covered by the actual source grid and has the
independent initial-input value after any complete program permutation. -/
theorem complete_tensor_readback (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) (m : Fin p.M) (n : Fin p.N) :
    completeInitialMemory plans p initial p.output (m.val*p.N+n.val) =
      Cell.mk .real (tensorValue plans p initial.mem m.val n.val (p.K/p.BK)) := by
  obtain ⟨pm,pn,i,j,hm,hn⟩ := complete_output_coverage p hBM hBN hM hN m n
  obtain ⟨hp,hpm,hpn⟩ := coordinates_cover p pm pn
  let lane : TileIndex [p.BM,p.BN] := (i,j,PUnit.unit)
  have hrow : row p (programState p initial initial.mem (pn.val*gridM p+pm.val)) i.val = m.val := by
    simpa [row, programState, hpm] using hm
  have hcol : col p (programState p initial initial.mem (pn.val*gridM p+pm.val)) j.val = n.val := by
    simpa [col, programState, hpn] using hn
  have h := scheduled_readback plans p initial (pn.val*gridM p+pm.val) hp lane
    (List.range (gridCount p)) initial.mem (fun k hk => List.mem_range.mp hk) (List.mem_range.mpr hp)
  rw [accumulated_tensor] at h
  simp only [outputOffset, lane] at h
  rw [hrow, hcol] at h
  exact h

#axiomsClean complete_tensor_readback
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
