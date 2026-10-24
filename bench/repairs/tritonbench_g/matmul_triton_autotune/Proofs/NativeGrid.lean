import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeExecution
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Refinement

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000

def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (b : Bases) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (execute M D r b pid s).bind (runPrograms M D r b rest)

theorem empty_grid (r : Request) (empty : r.rows=0 ∨ r.columns=0) : programs r=0 := by
  rcases empty with h | h
  all_goals cases hc : r.config <;> simp [programs,tileRows,tileColumns,h,hc,Config.bm,Config.bn]

theorem programs_correspondence {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (ids : List Nat) (valid : ∀ pid∈ids, pid<programs r ∧ pid<2^31)
    (s u : State α) (same : s.mem=u.mem) :
    ∃ native typed, runPrograms M D r (bases t) ids s=some native ∧
      Repair.runPrograms M D r ids u=some typed ∧ native.mem=typed.mem := by
  induction ids generalizing s u with
  | nil => exact ⟨s,u,rfl,rfl,same⟩
  | cons pid rest ih =>
    obtain ⟨hp,hpid⟩ := valid pid (by simp)
    obtain ⟨typed,he,hm⟩ := Source.execution_memory M D r.parameters (instanceState r pid u)
    have hmid : (FPValue.written M D r.parameters .single (instanceState r pid s)).mem=typed.mem := by
      rw [hm]
      exact written_mem_congr M D r.parameters _ _ same rfl
    obtain ⟨nf,tf,hn,ht,hfinal⟩ := ih (fun p h => valid p (List.mem_cons_of_mem pid h)) _ typed hmid
    refine ⟨nf,tf,?_,?_,hfinal⟩
    · rw [runPrograms,execute_eq M D r prepared t pid hp hpid s]
      exact hn
    · change (FP.Structural.exec (FPValue.backend M D) (Source.kernel r.parameters)
        (instanceState r pid u)).bind (Repair.runPrograms M D r rest)=some tf
      rw [he]
      exact ht

theorem host_correspondence {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (prepared : State α) (t : TensorAPI r prepared)
    (hgrid : programs r<2^31) (final : State α)
    (done : Repair.host M D r prepared=.ok (r.C,final)) :
    ∃ native, runPrograms M D r (bases t) (List.range (programs r)) prepared=some native ∧
      native.mem=final.mem := by
  obtain ⟨native,typed,hn,ht,hm⟩ := programs_correspondence M D r prepared t _
    (fun pid h => ⟨List.mem_range.mp h,lt_trans (List.mem_range.mp h) hgrid⟩) prepared prepared rfl
  have he : typed=final := by
    have hd := t.dimensions
    have hc := t.contiguousA
    simpa [Repair.host,hd,hc,ht] using done
  exact ⟨native,hn,by simpa [he] using hm⟩

/-- Complete byte safety is retained in the public wrapper postcondition,
alongside executable correspondence. No active mask hides an invalid pointer. -/
structure FullAccessSafety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) : Prop where
  metadata : ∀ pid, pid<programs r → ∀ n∈Native.metadata r, (scalar n).toInt=(n : Int)
  pid : ∀ pid, pid<programs r → (fromI32 pid).toInt=(pid : Int)
  control : ∀ pid, pid<programs r → Native.control r pid=naturalControl r pid
  divisors : ∀ pid, pid<programs r → 0<r.config.gm*tileColumns r ∧
    0<Whole.groupSize (tileRows r) (tileColumns r) r.config.gm pid
  inputs : ∀ pid k, pid<programs r → k<r.parameters.numKBlocks →
    (∀ i : Fin r.config.bm, ∀ e : Fin r.config.bk,
      FullBytes t.baseA 4 t.extentA (aPtr r t.baseA pid (word k) i.val e.val)) ∧
    (∀ e : Fin r.config.bk, ∀ j : Fin r.config.bn,
      FullBytes t.baseB 4 t.extentB (bPtr r t.baseB pid (word k) e.val j.val))
  outputs : ∀ pid, pid<programs r → ∀ i : Fin r.config.bm, ∀ j : Fin r.config.bn,
    FullBytes t.baseC 2 (r.rows*r.columns) (cPtr r t.baseC pid i.val j.val)
  backedge : ∀ pid k, pid<programs r → k<r.parameters.numKBlocks →
    (word k).toInt=(k : Int) ∧ (word k+word 1).toInt=((k+1 : Nat) : Int)

theorem full_access_safety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (hgrid : programs r<2^31) : FullAccessSafety r s t where
  metadata pid hp := metadata_exact r s t pid hp
  pid pid hp := by rw [fromI32_eq pid (lt_trans hp hgrid),signed pid (by omega)]
  control pid hp := control_eq r s t pid hp (lt_trans hp hgrid)
  divisors pid hp := ⟨(candidate_scalar_control_safe r pid hp).2.2.1,
    (candidate_scalar_control_safe r pid hp).2.2.2.1⟩
  inputs pid k hp hk :=
    ⟨fun i e => (a_access r s t pid k hp (lt_trans hp hgrid) hk i e).2.1,
      fun e j => (b_access r s t pid k hp (lt_trans hp hgrid) hk e j).2.1⟩
  outputs pid hp i j := (c_access r s t pid hp (lt_trans hp hgrid) i j).2.1
  backedge pid k hp hk := backedge_exact r s t pid k hp hk

#axiomsClean programs_correspondence
#axiomsClean host_correspondence
#axiomsClean full_access_safety
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
