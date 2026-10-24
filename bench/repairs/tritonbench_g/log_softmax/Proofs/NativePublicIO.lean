import bench.repairs.tritonbench_g.log_softmax.NativeIO
import bench.repairs.tritonbench_g.log_softmax.Proofs.IOBinding
import bench.repairs.tritonbench_g.log_softmax.Proofs.Contract

namespace VeriTile.Bench.Repairs.LogSoftmax.NativePublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

theorem exact_equivalence (direction : Direction) (R : Native.Reductions)
    (M N K BM : Nat) :
    (NativeIO.original direction R M N K BM).Equivalent
      (NativeIO.commuted direction R M N K BM) := by
  intro vals _ bounds bounded initial _
  have lsafe := IOBinding.schedule_safe direction id id
    (FP.ProfiledRewrite.engine (Native.algebra R)) Bool.false M N K BM _
    (block_positive N) bounds vals bounded (KernelGrid.programs M N K BM) initial
  have rsafe : ScheduleSafe bounds
      (IO.schedule direction (FP.ProfiledRewrite.engine (Native.algebra R))
        Bool.true M N K BM (Grid.blockColumns N)) initial := by
    cases direction
    · exact IOBinding.schedule_safe .forward id id
        (FP.ProfiledRewrite.engine (Native.algebra R)) Bool.true M N K BM _
        (block_positive N) bounds vals bounded _ initial
    · exact IOBinding.schedule_safe .backward id id
        (FP.ProfiledRewrite.engine (Native.algebra R)) Bool.true M N K BM _
        (block_positive N) bounds vals bounded _ initial
  obtain ⟨final,original,commuted,preserved⟩ := Observed.success_frame direction (Native.algebra R)
    M N K BM (Grid.blockColumns N) (block_positive N) (KernelGrid.programs M N K BM) initial
  refine ⟨?_,rsafe,final,final,?_,?_,?_,?_,?_⟩
  · cases direction <;> exact lsafe
  · cases direction <;>
      simpa only [NativeIO.original,IO.grid,IO.forward,IO.backward,IO.schedule,
        IOBinding.observed_run] using original
  · simpa only [NativeIO.commuted,IO.commuted,IO.schedule,IOBinding.observed_run] using commuted
  · intro o j _
    cases direction <;> rfl
  · exact IOBinding.frame direction id id (FP.ProfiledRewrite.engine (Native.algebra R))
      Bool.false M N K BM _ vals initial final preserved
  · exact IOBinding.frame direction id id (FP.ProfiledRewrite.engine (Native.algebra R))
      Bool.false M N K BM _ vals initial final preserved

theorem observed_equivalence (direction : Direction) (R : Native.Reductions)
    (M N K BM : Nat) (I : Native.Instructions) (S : Native.Schedule) :
    (NativeIO.original direction R M N K BM).ObservedEquivalent
      (NativeIO.commuted direction R M N K BM) (NativeIO.observation direction R M N K BM I S) :=
  ProgramIO.ObservedEquivalent.of_equivalent (exact_equivalence direction R M N K BM) _

end
end VeriTile.Bench.Repairs.LogSoftmax.NativePublicIO
