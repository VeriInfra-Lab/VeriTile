import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem affine_implements (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) :
    (affineIO M rows N wc bc pid bias).Implements (affineGraph M rows N pid) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    simp only [affineIO,scheduleSafe_singleton,Invocation.Safe,invocation]
    apply KernelSafety.affine
    · intro row hr j hj
      exact hb.2.1 ⟨0,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,j,PUnit.unit) hj
    · intro row hr j hj
      exact hb.2.1 ⟨1,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,j,PUnit.unit) hj
    · intro row hr
      exact hb.2.1 ⟨3,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,PUnit.unit) trivial
    · intro row hr
      exact hb.2.1 ⟨2,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.2.1 ⟨0,by simp [affineIO,affineOutputs]⟩ (j,PUnit.unit) hj
    · intro h j hj
      exact hb.2.2.1 ⟨1,by simp [affineIO,affineOutputs]⟩ (j,PUnit.unit) ⟨h,hj⟩
  · intro vals initial hin
    let u := (invocation M (KernelAffine.kernel rows N wc bc bias) pid).initial initial
    have hup : u.pids 0=pid := rfl
    have hum : u.mem=initial.mem := Invocation.initial_mem _ _
    have hc (j : Fin 128) : KernelAffine.carry M N pid u rows (j,PUnit.unit)=
        KernelReference.affineCarry M (decide (pid*128+j.val<N))
          (extend (fun row => affineX vals row j)) (extend (fun row => affineDY vals row j))
          (extend (affineMean vals)) (extend (affineInverse vals)) rows := by
      unfold KernelAffine.carry
      apply Congruence.affine
      · intro h row hr
        have active : pid*128+j.val<N := of_decide_eq_true h
        have hi := hin ⟨0,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,j,PUnit.unit) active
        simpa only [KernelAffine.raw,active,if_true,hum,extend,dif_pos hr] using hi
      · intro h row hr
        have active : pid*128+j.val<N := of_decide_eq_true h
        have hi := hin ⟨1,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,j,PUnit.unit) active
        simpa only [KernelAffine.raw,active,if_true,hum,extend,dif_pos hr] using hi
      · intro row hr
        simpa only [KernelAffine.saved,hum,extend,dif_pos hr] using
          hin ⟨2,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,PUnit.unit) trivial
      · intro row hr
        simpa only [KernelAffine.saved,hum,extend,dif_pos hr] using
          hin ⟨3,by simp [affineIO,affineInputs]⟩ (⟨row,hr⟩,PUnit.unit) trivial
    obtain ⟨final,ran,_,hw,hb,frame⟩ := KernelOutputs.affine_complete M rows N wc bc bias u
    rw [hup] at hw hb frame
    refine ⟨final,?_,?_,?_⟩
    · simpa only [affineIO,runSchedule_singleton,Invocation.run,KernelAffine.kernel] using ran
    · intro o j active
      have ho : o.val<2 := o.isLt
      rcases o with ⟨o,ho⟩
      interval_cases o
      · change mapCell id (final.mem "dw" (pid*128+j.1.val))= _
        rw [mapCell_identity,hw j.1 active,hc]
        rfl
      · change mapCell id (final.mem "db" (pid*128+j.1.val))= _
        rw [mapCell_identity,hb active.1 j.1 active.2,hc]
        rfl
    · intro r offset outside
      rw [← hum]
      apply frame
      intro inside
      rcases inside with ⟨rfl,j,hj,rfl⟩ | ⟨hb,rfl,j,hj,rfl⟩
      · exact (outside.1 ⟨0,by simp [affineIO,affineOutputs]⟩ (j,PUnit.unit) hj).elim
          (fun h => h rfl) (fun h => h rfl)
      · exact (outside.1 ⟨1,by simp [affineIO,affineOutputs]⟩ (j,PUnit.unit) ⟨hb,hj⟩).elim
          (fun h => h rfl) (fun h => h rfl)
end
end VeriTile.Bench.Repairs.FastLayernorm.Interface
