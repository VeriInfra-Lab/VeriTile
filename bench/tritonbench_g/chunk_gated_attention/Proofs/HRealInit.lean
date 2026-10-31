import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealLoop
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoads
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

def initial (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) : BlockState :=
  seedResult p r (decoded p u) a (preludeExecution p r u).val

theorem initialization_executes (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState)
    (hkey : (decoded p u).key*p.BK+p.BK≤p.K)
    (hvalue : (decoded p u).value*p.BV+p.BV≤p.V)
    (hm : ∀ off,u.readMem r.seed off=a off) :
    stepStmts (prelude p r++seedStage p r) u=some (initial p r a u) := by
  rw [stepStmts.append_some (preludeExecution p r u).property]
  exact seed_computes p r (decoded p u) a _ hkey hvalue
    (prelude_head p r u) (prelude_key p r u) (prelude_value p r u) (by simpa [BlockState.readMem] using hm)

theorem initial_integer (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) (name : RegName)
    (hb : name≠"b_h") (hp : name≠"p_h") :
    (initial p r a u).regs .int [] name=(preludeExecution p r u).val.regs .int [] name := by
  unfold initial seedResult
  by_cases h:p.hasSeed <;> simp only [h,Bool.false_eq_true,↓reduceIte,BlockState.setReg,hb,hp,↓reduceDIte]

theorem initial_memory (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) :
    (initial p r a u).mem=u.mem := by
  unfold initial seedResult
  by_cases h:p.hasSeed <;> simp only [h,Bool.false_eq_true,↓reduceIte] <;> rfl

theorem initial_pids (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) :
    (initial p r a u).pids=u.pids := by
  unfold initial seedResult
  by_cases h:p.hasSeed <;> simp only [h,Bool.false_eq_true,↓reduceIte] <;> rfl

theorem initial_undef (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) :
    (initial p r a u).undef=u.undef := by
  unfold initial seedResult
  by_cases h:p.hasSeed <;> simp only [h,Bool.false_eq_true,↓reduceIte] <;> rfl

theorem initial_carry (p : Parameters) (r : Regions) (a : Inputs) (u : BlockState) :
    (initial p r a.seed u).regs .real [p.BK,p.BV] "b_h"=
      some (stateTile p.toLayout (expected p.toLayout (decoded p u) a p.gateK p.hasSeed 0)) := by
  rw [←RealHInit.starting_expected]
  unfold initial seedResult RealHInit.starting
  by_cases h:p.hasSeed <;> simp [h]

theorem initial_strides (p : Parameters) (r : Regions) (a : Nat→ℝ) (u : BlockState) :
    StrideRegs p (initial p r a u) := by
  simpa [StrideRegs,initial_integer] using prelude_strides p r u

theorem invariant (p : Parameters) (r : Regions) (a : Inputs) (u : BlockState)
    (hk : ∀ off,u.readMem r.key off=a.key off)
    (hv : ∀ off,u.readMem r.value off=a.value off)
    (hg : ∀ off,u.readMem r.gate off=a.gate off) :
    Loop.Holds u p (decoded p u) r a 0 (initial p r a.seed u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,initial_carry p r a u,?_,?_,?_,?_,?_,
    initial_pids p r a.seed u,initial_undef p r a.seed u,initial_strides p r a.seed u⟩
  · simpa [initial_integer] using prelude_key p r u
  · simpa [initial_integer] using prelude_value p r u
  · simpa [initial_integer] using prelude_head p r u
  · intro off;simpa only [BlockState.readMem,initial_memory] using hk off
  · intro off;simpa only [BlockState.readMem,initial_memory] using hv off
  · intro off;simpa only [BlockState.readMem,initial_memory] using hg off
  · intro q _ _ hq;omega
  · intro _ _ _;rw [initial_memory]

#axiomsClean initialization_executes
#axiomsClean invariant
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
