import bench.tritonbench_g.fast_layernorm.Proofs.IOReference
import bench.tritonbench_g.fast_layernorm.Proofs.KernelSafety
import bench.tritonbench_g.fast_layernorm.Proofs.KernelOutputs

namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

@[simp] theorem mapCell_identity (cell : Cell α) : mapCell id cell=cell := by
  cases cell with | mk d v => cases d <;> rfl

@[simp] theorem invocation_pid (M : Algebra α) (kernel : ComputeKernel) (pid : Nat) (s : State α) :
    ((invocation M kernel pid).initial s).pids 0=pid := rfl

theorem forward_implements (M : Algebra α) (N B pid : Nat) (eps : ℝ) (bias : Bool) :
    (forwardIO M N B pid eps bias).Implements (forwardGraph M N B eps bias) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    simp only [forwardIO,scheduleSafe_singleton,Invocation.Safe,invocation]
    apply KernelSafety.forward
    · intro j hj
      exact hb.2.1 ⟨0,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨1,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) hj
    · intro h j hj
      exact hb.2.1 ⟨2,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) ⟨h,hj⟩
    · intro j hj
      exact hb.2.2.1 ⟨0,by simp [forwardIO,forwardOutputs]⟩ (j,PUnit.unit) hj
    · exact hb.2.2.1 ⟨1,by simp [forwardIO,forwardOutputs]⟩ PUnit.unit trivial
    · exact hb.2.2.1 ⟨2,by simp [forwardIO,forwardOutputs]⟩ PUnit.unit trivial
  · intro vals initial hin
    let u := (invocation M (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) pid).initial initial
    have hup : u.pids 0=pid := rfl
    have hum : u.mem=initial.mem := Invocation.initial_mem _ _
    have hx (j : Fin B) (hj : j.val<N) : KernelRows.raw u "x" N B j=forwardX vals j := by
      exact hin ⟨0,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) hj
    have hw (j : Fin B) (hj : j.val<N) : KernelRows.raw u "w" 0 B j=forwardW vals j := by
      simpa only [KernelRows.raw,Nat.mul_zero,Nat.zero_add,hum] using
        hin ⟨1,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) hj
    have hb (h : bias=Bool.true) (j : Fin B) (hj : j.val<N) : KernelRows.raw u "b" 0 B j=forwardB vals j := by
      simpa only [KernelRows.raw,Nat.mul_zero,Nat.zero_add,hum] using
        hin ⟨2,by simp [forwardIO,forwardInputs]⟩ (j,PUnit.unit) ⟨h,hj⟩
    obtain ⟨final,ran,_,hy,hr,hm,frame⟩ := KernelOutputs.forward_complete M N B eps bias u
    refine ⟨final,?_,?_,?_⟩
    · simpa only [forwardIO,runSchedule_singleton,Invocation.run] using ran
    · intro o j active
      have ho : o.val<3 := o.isLt
      rcases o with ⟨o,ho⟩
      interval_cases o
      · change mapCell id (final.mem "y" (pid*N+j.1.val))= _
        rw [mapCell_identity,← hup,hy j.1 active,
          congrFun (Congruence.output M N eps bias _ _ _ _ _ _ hx hw hb) j.1]
        rfl
      · change mapCell id (final.mem "r" pid)= _
        rw [mapCell_identity,← hup,hr]
        simp only [KernelReference.savedInverse,Congruence.inverse M N eps _ _ hx]
        rfl
      · change mapCell id (final.mem "mu" pid)= _
        rw [mapCell_identity,← hup,hm]
        simp only [KernelReference.savedMean,Congruence.mean M N _ _ hx]
        rfl
    · intro r offset outside
      rw [← hum]
      apply frame
      intro inside
      rcases inside with ⟨rfl,j,hj,rfl⟩ | ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩
      · exact (outside.1 ⟨0,by simp [forwardIO,forwardOutputs]⟩ (j,PUnit.unit) hj).elim
          (fun h => h rfl) (fun h => h (congrArg (fun row => row*N+j.val) hup))
      · exact (outside.1 ⟨1,by simp [forwardIO,forwardOutputs]⟩ PUnit.unit trivial).elim
          (fun h => h rfl) (fun h => h hup)
      · exact (outside.1 ⟨2,by simp [forwardIO,forwardOutputs]⟩ PUnit.unit trivial).elim
          (fun h => h rfl) (fun h => h hup)

theorem backward_implements (M : Algebra α) (N B pid : Nat) :
    (backwardIO M N B pid).Implements (backwardGraph M N B) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    simp only [backwardIO,scheduleSafe_singleton,Invocation.Safe,invocation]
    apply KernelSafety.backward
    · intro j hj
      exact hb.2.1 ⟨0,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨1,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    · intro j hj
      exact hb.2.1 ⟨2,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    · intro j hj
      exact hb.2.2.1 ⟨0,by simp [backwardIO,backwardOutputs]⟩ (j,PUnit.unit) hj
    · exact hb.2.1 ⟨4,by simp [backwardIO,backwardInputs]⟩ PUnit.unit trivial
    · exact hb.2.1 ⟨3,by simp [backwardIO,backwardInputs]⟩ PUnit.unit trivial
  · intro vals initial hin
    let u := (invocation M (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) pid).initial initial
    have hup : u.pids 0=pid := rfl
    have hum : u.mem=initial.mem := Invocation.initial_mem _ _
    have hx (j : Fin B) (hj : j.val<N) : KernelRows.raw u "x" N B j=backwardX vals j := by
      exact hin ⟨0,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    have hdy (j : Fin B) (hj : j.val<N) : KernelRows.raw u "dy" N B j=backwardDY vals j := by
      exact hin ⟨1,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    have hw (j : Fin B) (hj : j.val<N) : KernelRows.raw u "w" 0 B j=backwardW vals j := by
      simpa only [KernelRows.raw,Nat.mul_zero,Nat.zero_add,hum] using
        hin ⟨2,by simp [backwardIO,backwardInputs]⟩ (j,PUnit.unit) hj
    have hm : (u.mem "mu" (u.pids 0)).read .real=backwardMean vals :=
      hin ⟨3,by simp [backwardIO,backwardInputs]⟩ PUnit.unit trivial
    have hr : (u.mem "r" (u.pids 0)).read .real=backwardInverse vals :=
      hin ⟨4,by simp [backwardIO,backwardInputs]⟩ PUnit.unit trivial
    obtain ⟨final,ran,_,hy,frame⟩ := KernelOutputs.backward_complete M N B u
    refine ⟨final,?_,?_,?_⟩
    · simpa only [backwardIO,runSchedule_singleton,Invocation.run] using ran
    · intro o j active
      have ho : o.val<1 := o.isLt
      rcases o with ⟨o,ho⟩
      interval_cases o
      change mapCell id (final.mem "dx" (pid*N+j.1.val))= _
      rw [mapCell_identity,← hup,hy j.1 active,hm,hr,
        congrFun (Congruence.gradient M N _ _ _ _ _ _ _ _ hx hdy hw) j.1]
      rfl
    · intro r offset outside
      rw [← hum]
      apply frame
      rintro ⟨rfl,j,hj,rfl⟩
      exact (outside.1 ⟨0,by simp [backwardIO,backwardOutputs]⟩ (j,PUnit.unit) hj).elim
        (fun h => h rfl) (fun h => h (congrArg (fun row => row*N+j.val) hup))

end
end VeriTile.Bench.Repairs.FastLayernorm.Interface
