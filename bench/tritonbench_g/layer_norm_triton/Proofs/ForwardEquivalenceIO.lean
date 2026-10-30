import bench.tritonbench_g.layer_norm_triton.ForwardCandidateIO
import bench.tritonbench_g.layer_norm_triton.Proofs.ForwardIO
import bench.tritonbench_g.layer_norm_triton.Proofs.ForwardCandidateSafety

namespace VeriTile.Bench.Repairs.LayerNormTriton.ForwardEquivalenceIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem equivalent (m : Model α) (stride N B row : Nat) (eps : ℝ)
    (observe : RegionName → Nat → α → α) (positive : 0<N) (covers : N≤B) :
    (Forward.io m stride N B row eps observe).Equivalent
      (candidate m stride N B row eps observe) := by
  intro vals _ bounds boundsOK initial binding
  obtain ⟨safe,t,ran,values,frame⟩ :=
    ForwardIO.implements m stride N B row eps observe positive covers vals trivial bounds boundsOK initial binding
  have csafe : ScheduleSafe bounds ((candidate m stride N B row eps observe).schedule vals) initial := by
    change ScheduleSafe bounds [job m (SourceCandidate.forward stride N B eps) row] initial
    rw [scheduleSafe_singleton]
    apply SourceSafety.forward_candidate_safe m stride N B eps positive covers
    · intro j hj
      exact boundsOK.2.1 (0 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.1 (1 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.1 (2 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.2.1 (0 : Fin 3) (⟨j,hj⟩,()) trivial
    · exact boundsOK.2.2.1 (1 : Fin 3) () trivial
    · exact boundsOK.2.2.1 (2 : Fin 3) () trivial
  have eq : runSchedule ((candidate m stride N B row eps observe).schedule vals) initial=
      runSchedule ((Forward.io m stride N B row eps observe).schedule vals) initial := by
    simp only [candidate,Forward.io,runSchedule_singleton,Invocation.run,job,Invocation.initial]
    exact SourceCandidate.forward_equiv _ _ _ _ _ positive covers _
  exact ⟨safe,csafe,t,t,ran,eq.trans ran,fun _ _ _ => rfl,frame,frame⟩
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayerNormTriton.ForwardEquivalenceIO
