import bench.repairs.tritonbench_g.mixed_sparse_attention.PairIO
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.FusedKernel

namespace VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem invocation_run (M : Algebra α) (D : Dot α) (c : Config) (p : Program) (s : State α) :
    (KernelIO.invocation M D c p).run s=(invocation M D c p).run s :=
  execution_equal (backend M D) c _

theorem schedule_run (M : Algebra α) (D : Dot α) (c : Config) (work : List Program) (s : State α) :
    runSchedule (work.map (KernelIO.invocation M D c)) s=
      runSchedule (work.map (invocation M D c)) s := by
  induction work generalizing s with
  | nil=>rfl
  | cons p rest ih=>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation M D c p).run s <;> simp [Option.bind,ih]

theorem schedule_safe (M : Algebra α) (D : Dot α) (c : Config) (work : List Program)
    (cap : RegionName → Nat) (s : State α)
    (h : ScheduleSafe cap (work.map (KernelIO.invocation M D c)) s) :
    ScheduleSafe cap (work.map (invocation M D c)) s := by
  induction work generalizing s with
  | nil=>trivial
  | cons p rest ih=>
    refine ⟨(safety_equal (backend M D) c cap _).mp h.1,?_⟩
    intro next ran
    apply ih next
    apply h.2 next
    exact (invocation_run M D c p s).trans ran

theorem equivalent (M : Algebra α) (D : Dot α) (c : Config)
    (sep : Separated c) (layout : Contiguous c) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    (KernelIO.io M D c work).EquivalentOn (variant M D c work) (KernelIO.domain c) := by
  intro xs hd cap hb initial hin
  have reads:=KernelIO.input_reads M D c work xs initial hin
  obtain ⟨safe,final,ran,_,frame⟩:=KernelIO.whole M D c sep layout hbm hbn work order xs hd initial reads
  refine ⟨safe cap hb,schedule_safe M D c work cap initial (safe cap hb),final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run M D c work initial).symm.trans ran
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
