import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SourcePair
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.AffineDirectSafety
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOAffine
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.FastLayernorm.IOPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_pair (M : Algebra α) (N B pid : Nat) (eps : ℝ) (bias : Bool) :
    (io M (.forward N B pid eps bias)).Equivalent (variant M (.forward N B pid eps bias)) := by
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := forward_implements M N B pid eps bias values trivial bounds hb s hin
  have he := SourcePair.forward_exec (engine M)
    ((invocation M (SourcePair.forward Bool.false N B eps bias) pid).initial s) N B eps bias
  have hs := SourcePair.forward_safe (engine M)
    ((invocation M (SourcePair.forward Bool.false N B eps bias) pid).initial s) bounds N B eps bias
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · simpa only [variant,scheduleSafe_singleton,Invocation.Safe] using
      hs.mp (by simpa only [forwardIO,scheduleSafe_singleton,Invocation.Safe] using safe)
  · have ran' := ran
    simp only [forwardIO,runSchedule_singleton,Invocation.run] at ran'
    simpa only [variant,runSchedule_singleton,Invocation.run] using he.symm.trans ran' 
  · intro o j active; rfl

theorem backward_pair (M : Algebra α) (N B pid : Nat) :
    (io M (.backward N B pid)).Equivalent (variant M (.backward N B pid)) := by
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := backward_implements M N B pid values trivial bounds hb s hin
  have he := SourcePair.backward_exec (engine M)
    ((invocation M (SourcePair.backward Bool.false N B) pid).initial s) N B
  have hs := SourcePair.backward_safe (engine M)
    ((invocation M (SourcePair.backward Bool.false N B) pid).initial s) bounds N B
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · simpa only [variant,scheduleSafe_singleton,Invocation.Safe] using
      hs.mp (by simpa only [backwardIO,scheduleSafe_singleton,Invocation.Safe] using safe)
  · have ran' := ran
    simp only [backwardIO,runSchedule_singleton,Invocation.run] at ran'
    simpa only [variant,runSchedule_singleton,Invocation.run] using he.symm.trans ran' 
  · intro o j active; rfl

theorem affine_pair (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool) :
    (io M (.affine rows N wc bc pid bias)).Equivalent (variant M (.affine rows N wc bc pid bias)) := by
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := affine_implements M rows N wc bc pid bias values trivial bounds hb s hin
  let u := (invocation M (KernelAffine.kernel rows N wc bc bias) pid).initial s
  obtain ⟨left,hl,hml⟩ := KernelAffine.source_memory M rows N wc bc bias u
  obtain ⟨right,hr,hmr⟩ := AffineDirect.source_memory M rows N wc bc bias u
  have same : right.mem = left.mem := hmr.trans hml.symm
  have left_eq : final = left := by
    apply Option.some.inj
    have ran' := ran
    simp only [affineIO,runSchedule_singleton,Invocation.run] at ran'
    exact ran'.symm.trans hl
  subst left
  refine ⟨safe,?_,final,right,ran,?_,?_,frame,?_⟩
  · change ScheduleSafe bounds [invocation M (AffineDirect.kernel rows N wc bc bias) pid] s
    rw [scheduleSafe_singleton]
    apply AffineDirectSafety.affine
    · intro row hrow j hj
      exact hb.2.1 ⟨0,by simp [io,affineIO,affineInputs]⟩ (⟨row,hrow⟩,j,PUnit.unit) hj
    · intro row hrow j hj
      exact hb.2.1 ⟨1,by simp [io,affineIO,affineInputs]⟩ (⟨row,hrow⟩,j,PUnit.unit) hj
    · intro row hrow
      exact hb.2.1 ⟨3,by simp [io,affineIO,affineInputs]⟩ (⟨row,hrow⟩,PUnit.unit) trivial
    · intro row hrow
      exact hb.2.1 ⟨2,by simp [io,affineIO,affineInputs]⟩ (⟨row,hrow⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.2.1 ⟨0,by simp [io,affineIO,affineOutputs]⟩ (j,PUnit.unit) hj
    · intro h j hj
      exact hb.2.2.1 ⟨1,by simp [io,affineIO,affineOutputs]⟩ (j,PUnit.unit) ⟨h,hj⟩
  · simpa only [variant,runSchedule_singleton,Invocation.run] using hr
  · intro o j active
    change mapCell id (final.mem _ _) = mapCell id (right.mem _ _)
    rw [same]
  · intro region off outside
    rw [same]
    exact frame region off outside

theorem equivalent (M : Algebra α) (stage : Stage) :
    (io M stage).Equivalent (variant M stage) := by
  cases stage with
  | forward N B pid eps bias => exact forward_pair M N B pid eps bias
  | backward N B pid => exact backward_pair M N B pid
  | affine rows N wc bc pid bias => exact affine_pair M rows N wc bc pid bias
end
end VeriTile.Bench.Repairs.FastLayernorm.IOPair
