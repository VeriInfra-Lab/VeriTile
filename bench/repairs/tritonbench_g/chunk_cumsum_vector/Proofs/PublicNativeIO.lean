import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.NativeInputBinding
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPAccessSource

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem bounds_binding (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (xs : (io M dot h c p).Inputs) (bounds : RegionBounds)
    (bounded : (io M dot h c p).InBounds bounds xs) : AccessBounds.Bounds h p bounds := by
  intro time j ht hj
  exact ⟨bounded.2.1 (⟨0,by decide⟩ : Fin 1) (⟨time,ht⟩,j,PUnit.unit) hj,
    bounded.2.2.1 (⟨0,by decide⟩ : Fin 1) (⟨time,ht⟩,j,PUnit.unit) hj⟩

theorem frame_binding (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (xs : (io M dot h c p).Inputs) (u v : State α)
    (frame : FPMemory.Frame h p u v) : (io M dot h c p).Frame xs u v := by
  intro r offset outside
  symm
  apply frame r offset
  rintro ⟨same,time,j,ht,hj,whereAt⟩
  have miss := outside.1 (⟨0,by decide⟩ : Fin 1) (⟨time,ht⟩,j,PUnit.unit) hj
  exact miss.elim (fun no => no same) (fun no => no whereAt)

theorem equivalent (M : Algebra α) (dot : DotBackend α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (chosen : c∈configurations)
    (p : Geometry.Program h) :
    (io M dot h c p).EquivalentOn (candidate M dot h c p)
      (fun xs => finiteCarry M D h c p (xs (⟨0,by decide⟩ : Fin 1))) := by
  apply ProgramIO.EquivalentOn.intro
  · intro xs _ bounds bounded s _
    rw [scheduleSafe_singleton]
    change TraceSafe (machine M dot) bounds (Kernels.source h c)
      ((invocation M dot h c p Bool.false).initial s)
    exact FPAccess.source_safe M dot bounds h layout c
      (Kernels.configurations_positive c chosen).1 p (bounds_binding M dot h c p xs bounds bounded) _
      (by simp [invocation,Invocation.initial,bindRegisters])
      (by simp [invocation,Invocation.initial,bindRegisters])
  · intro xs _ bounds bounded s _
    change ScheduleSafe bounds [invocation M dot h c p Bool.true] s
    rw [scheduleSafe_singleton]
    change TraceSafe (machine M dot) bounds (Kernels.sumLeft h c)
      ((invocation M dot h c p Bool.true).initial s)
    exact FPAccess.candidate_safe M dot bounds h layout c
      (Kernels.configurations_positive c chosen).1 p (bounds_binding M dot h c p xs bounds bounded) _
      (by simp [invocation,Invocation.initial,bindRegisters])
      (by simp [invocation,Invocation.initial,bindRegisters])
  · intro xs domain _ _ s binding
    let start := (invocation M dot h c p Bool.false).initial s
    have pinned : (io M dot h c p).InputBinding xs start := by
      simpa only [ProgramIO.InputBinding,Invocation.initial_mem] using binding
    obtain ⟨a,b,ranA,ranB,equal,frameA,frameB⟩ := Rewrite.successful_pair M dot D model h layout c
      (Kernels.configurations_positive c chosen).1 p start
      (by simp [start,invocation,Invocation.initial,bindRegisters])
      (by simp [start,invocation,Invocation.initial,bindRegisters])
      (finite_binding M dot D h c p xs start pinned domain)
    refine ⟨a,b,?_,?_,?_,?_,?_⟩
    · simpa only [runSchedule_singleton] using ranA
    · simpa only [candidate, runSchedule_singleton] using ranB
    · intro o j active
      exact congrArg (mapCell id) (congrFun (congrFun equal _) _)
    · exact frame_binding M dot h c p xs start a frameA
    · exact frame_binding M dot h c p xs start b frameB

#specNonCircular finiteCarry avoiding [Kernels.source,Kernels.sumLeft,run,FPExecution.source_correct]
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
