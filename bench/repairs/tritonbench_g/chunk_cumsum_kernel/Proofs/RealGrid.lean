import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealComplete

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealGrid
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

def atRow (rows pid : Nat) (s : BlockState) : BlockState :=
  { s with regs := fun _ _ _ => none
           pids := fun ax => if ax=0 then pid else 0
           numPids := fun ax => if ax=0 then rows else 1 }

def run (input output : RegionName) (T rows : Nat) (cfg : Config) :
    List Nat → BlockState → Option BlockState
  | [], s => some s
  | pid::rest, s => (exec (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType.toAlgKernel
      (atRow rows pid s)).bind (run input output T rows cfg rest)

def expected (input : RegionName) (T pid i : Nat) (s : BlockState) : ℝ :=
  ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), s.readMem input (pid*T+j)

def Frame (output : RegionName) (T : Nat) (order : List Nat) (s t : BlockState) : Prop :=
  ∀ rg off, (rg≠output ∨ ∀ pid∈order, ∀ i, i<T → off≠pid*T+i) → t.mem rg off=s.mem rg off

theorem row_unique {T p q i j : Nat} (hi : i<T) (hj : j<T) (h : p*T+i=q*T+j) : p=q := by
  have positive : 0<T := by omega
  have quotients := congrArg (fun n => n/T) h
  simpa [Nat.mul_comm p T,Nat.mul_comm q T,Nat.mul_add_div positive,
    Nat.div_eq_of_lt hi,Nat.div_eq_of_lt hj] using quotients

/-- Actual complete kernels execute successively on evolving memory.
Arbitrary finite orders (including repeated autotune row work) are allowed. -/
theorem complete (input output : RegionName) (T rows : Nat)
    (cfg : Config) (separate : output≠input) (order : List Nat) (s : BlockState) :
    ∃ t, run input output T rows cfg order s=some t ∧
      (∀ pid∈order, ∀ i, i<T → t.readMem output (pid*T+i)=expected input T pid i s) ∧
      Frame output T order s t := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,by simp,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,hm,hvalue,hframe⟩ := RealIndexBridge.complete input output T cfg
      (atRow rows pid s) separate
    obtain ⟨t,ht,values,frame⟩ := ih mid
    have inputMem : mid.mem input=s.mem input := by
      funext off
      exact hframe input off (Or.inl (Ne.symm separate))
    have expected_same (p i : Nat) : expected input T p i mid=expected input T p i s := by
      simp only [expected,BlockState.readMem,inputMem]
    refine ⟨t,?_,?_,?_⟩
    · simp only [run,hm,Option.bind_some]
      exact ht
    · intro p member i hi
      by_cases hr : p∈rest
      · exact (values p hr i hi).trans (expected_same p i)
      · have hp : p=pid := by simpa [hr] using member
        subst p
        have same : t.mem output (pid*T+i)=mid.mem output (pid*T+i) := by
          apply frame output (pid*T+i) (Or.inr ?_)
          intro q hq j hj he
          exact hr ((row_unique hi hj he) ▸ hq)
        rw [BlockState.readMem,same]
        simpa [atRow,expected,BlockState.readMem] using hvalue i hi
    · intro rg off h
      have tailFrame : rg≠output ∨ ∀ p∈rest, ∀ i, i<T → off≠p*T+i := by
        rcases h with hn | hn
        · exact Or.inl hn
        · exact Or.inr (fun p hp i hi => hn p (List.mem_cons_of_mem _ hp) i hi)
      rw [frame rg off tailFrame]
      apply hframe rg off
      rcases h with hn | hn
      · exact Or.inl hn
      · exact Or.inr (fun i hi => by simpa [atRow] using hn pid (List.mem_cons_self) i hi)

#axiomsClean row_unique
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealGrid
