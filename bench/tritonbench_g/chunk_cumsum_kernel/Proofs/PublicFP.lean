import bench.tritonbench_g.chunk_cumsum_kernel.IO
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000

variable {α : Type} [Inhabited α]

private theorem safe (ops : FPRow.Ops α) (v : Bool) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) (vals : (io ops input output T cfg pid).Inputs)
    (bounds : RegionName → Nat) (hb : (io ops input output T cfg pid).InBounds bounds vals)
    (s : State α) :
    TraceSafe (engine ops.algebra) bounds (SourceSafety.source v input output T cfg.block) (start pid s) := by
  apply SourceSafety.trace_safe
  · intro j hj
    exact hb.2.1 0 (⟨j,hj⟩,()) trivial
  · intro j hj
    exact hb.2.2.1 0 (⟨j,hj⟩,()) trivial

private theorem frame (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) (vals : (io ops input output T cfg pid).Inputs)
    (s t : State α) (h : FPRow.Frame output pid T (start pid s) t) :
    (io ops input output T cfg pid).Frame vals s t := by
  intro r address outside
  apply h
  by_cases hr : r=output
  · right
    intro j hj
    exact (outside.1 0 (⟨j,hj⟩,()) trivial).resolve_left (fun h => h hr)
  · exact Or.inl hr

/-- Shared IO equivalence reuses the existing complete two-source execution.
The only numerical domain is the independently computed, subsequently used
carry pair; actual trace safety is proved separately for every input. -/
theorem equivalent (ops : FPRow.Ops α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat) :
    (io ops input output T cfg pid).EquivalentOn (candidate ops input output T cfg pid)
      (domain ops D input output T cfg pid) := by
  intro vals numerical bounds hb s hin
  have reads : ∀ j,j<T → ((start pid s).mem input ((start pid s).pids 0*T+j)).read .fp32 = row (α := α) (vals 0) j := by
    intro j hj
    simpa [row,hj] using hin 0 (⟨j,hj⟩,()) trivial
  have finite := CarryDomain.of_inputs ops D input output T cfg (row (α := α) (vals 0)) (start pid s) reads numerical
  obtain ⟨a,b,ha,hb',equal,fa,fb⟩ := Rewrite.successful_pair ops D model input output T cfg.block (start pid s) finite
  refine ⟨?_,?_,a,b,?_,?_,?_,frame ops input output T cfg pid vals s a fa,
    frame ops input output T cfg pid vals s b fb⟩
  · rw [scheduleSafe_singleton]
    exact safe ops Bool.false input output T cfg pid vals bounds hb s
  · rw [scheduleSafe_singleton]
    exact safe ops Bool.true input output T cfg pid vals bounds hb s
  · rw [runSchedule_singleton]
    exact ha
  · rw [runSchedule_singleton]
    exact hb'
  · intro o j _
    rw [equal]

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicFP
