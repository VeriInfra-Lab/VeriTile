import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicData
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyLoops

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem tile_dimensions (p : Core.Parameters) (tile : Nat) (ht : tile<p.problem.tiles) :
    0<p.problem.M ∧ 0<p.problem.N := by
  constructor
  · by_contra h
    have hz : p.problem.M=0 := by omega
    simp [Problem.tiles,Problem.rows,RealMath.blocks,hz] at ht
  · by_contra h
    have hz : p.problem.N=0 := by omega
    simp [Problem.tiles,Problem.columns,RealMath.blocks,hz] at ht

theorem tile_bounds (P : Source.Arithmetic α α) (p : Core.Parameters)
    (xs : (io P p).Inputs) (bounds : RegionName → Nat) (hb : (io P p).InBounds bounds xs)
    (tile : Nat) (ht : tile<p.problem.tiles) : SourceSafety.TileBounds p bounds tile := by
  have hd := tile_dimensions p tile ht
  constructor
  · intro q i hk
    have hr := clamp_bound p.problem.M (p.problem.row tile i.1.val) hd.1
    have h := hb.2.1 0 (⟨_,hr⟩,⟨_,hk⟩,PUnit.unit) trivial
    change clamp p.problem.M (p.problem.row tile i.1.val)*p.am+p.problem.k q i.2.1.val*p.ak<bounds "A" at h
    simpa [Core.aOffset,clamp,hk] using h
  · intro q i hk
    have hc := clamp_bound p.problem.N (p.problem.column tile i.2.1.val) hd.2
    have h := hb.2.1 1 (⟨_,hk⟩,⟨_,hc⟩,PUnit.unit) trivial
    change p.problem.k q i.1.val*p.bk+clamp p.problem.N (p.problem.column tile i.2.1.val)*p.bn<bounds "B" at h
    simpa [Core.bOffset,clamp,hk] using h
  · intro i hi
    have h := hb.2.2.1 0 (⟨_,hi.1⟩,⟨_,hi.2⟩,PUnit.unit) trivial
    change p.problem.row tile i.1.val*p.cm+p.problem.column tile i.2.1.val*p.cn<bounds "C" at h
    simpa [Core.cOffset,clamp,hi.1,hi.2,Nat.mul_comm] using h

theorem job_run (P : Source.Arithmetic α α) (p : Core.Parameters) (pid : Nat) (s : State α) :
    (job P p pid).run s=Core.program P p pid s := by
  rfl

theorem schedule_run (P : Source.Arithmetic α α) (p : Core.Parameters) (pids : List Nat) (s : State α) :
    runSchedule (pids.map (job P p)) s=GridExecution.sequence (Core.program P p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih => simp [runSchedule,GridExecution.sequence,job_run,ih]

theorem schedule_safe (P : Source.Arithmetic α α) (p : Core.Parameters) (sms : 0<p.problem.SMS)
    (xs : (io P p).Inputs) (bounds : RegionName → Nat) (hb : (io P p).InBounds bounds xs)
    (initial : State α) : ScheduleSafe bounds ((io P p).schedule xs) initial := by
  have aux (pids : List Nat) (hpid : ∀ pid∈pids,pid<p.problem.grid) (s : State α) :
      ScheduleSafe bounds (pids.map (job P p)) s := by
    induction pids generalizing s with
    | nil => trivial
    | cons pid rest ih =>
      refine ⟨?_,?_⟩
      · exact SourceSafety.program_safe P p s pid sms (hpid pid (by simp)) bounds (tile_bounds P p xs bounds hb)
      · intro t _
        exact ih (fun q hq => hpid q (by simp [hq])) t
  exact aux _ (fun pid hp => List.mem_range.mp hp) initial

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
