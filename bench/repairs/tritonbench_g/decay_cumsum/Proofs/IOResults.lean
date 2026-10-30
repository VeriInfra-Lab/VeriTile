import bench.repairs.tritonbench_g.decay_cumsum.IO

namespace VeriTile.Bench.Repairs.DecayCumsum.DecayIO
open VeriTile.Triton FP.Structural Geometry
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem input_reads (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) (vals : (io M kind h p prog scale).Inputs)
    (s : State α) (hin : (io M kind h p prog scale).InputBinding vals s)
    (r : RegionName) (hr : r∈Roles.inputs kind) (n : Nat) (hn : n<p.span) (j : Fin h.BK) :
    (s.mem r (FPPrefix.off h p prog n j)).read .real=logicalRead kind h p vals r n j := by
  cases kind
  case forward =>
    simp only [Roles.inputs,List.mem_cons,List.not_mem_nil,or_false] at hr
    subst r
    simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
      hin (0 : Fin 1) (⟨n,hn⟩,j,()) trivial
  case prepare =>
    simp only [Roles.inputs,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (0 : Fin 3) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (1 : Fin 3) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (2 : Fin 3) (⟨n,hn⟩,j,()) trivial
  case backward =>
    simp only [Roles.inputs,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (0 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (1 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (2 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (3 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (4 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (5 : Fin 7) (⟨n,hn⟩,j,()) trivial
    · simpa [io,memoryIO,inputs,inputIndex,logicalRead,hn,address,FPPrefix.off,mapValue] using
        hin (6 : Fin 7) (⟨n,hn⟩,j,()) trivial

theorem raw_frame (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) (vals : (io M kind h p prog scale).Inputs)
    (s final : State α) (frame : FPMemory.Frame h p prog (FPTile.outputs kind) s final) :
    (io M kind h p prog scale).Frame vals s final := by
  intro r address outside
  apply frame r address
  rintro ⟨member,t,j,same⟩
  cases kind
  case forward =>
    simp only [FPTile.outputs,FPBackwardBase.outputRegions,List.mem_cons,List.not_mem_nil,or_false] at member
    subst r
    have no := outside.1 (0 : Fin 1) (t,j,()) trivial
    change ("go" : RegionName)≠"go" ∨ address≠FPPrefix.off h p prog t.val j at no
    exact no.elim (fun ne => ne rfl) (fun ne => ne same)
  case prepare =>
    simp only [FPTile.outputs,FPBackwardBase.outputRegions,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have no := outside.1 (0 : Fin 2) (t,j,()) trivial
      change ("qg" : RegionName)≠"qg" ∨ address≠FPPrefix.off h p prog t.val j at no
      exact no.elim (fun ne => ne rfl) (fun ne => ne same)
    · have no := outside.1 (1 : Fin 2) (t,j,()) trivial
      change ("kg" : RegionName)≠"kg" ∨ address≠FPPrefix.off h p prog t.val j at no
      exact no.elim (fun ne => ne rfl) (fun ne => ne same)
  case backward =>
    simp only [FPTile.outputs,FPBackwardBase.outputRegions,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl
    · have no := outside.1 (0 : Fin 3) (t,j,()) trivial
      change ("dqo" : RegionName)≠"dqo" ∨ address≠FPPrefix.off h p prog t.val j at no
      exact no.elim (fun ne => ne rfl) (fun ne => ne same)
    · have no := outside.1 (1 : Fin 3) (t,j,()) trivial
      change ("dko" : RegionName)≠"dko" ∨ address≠FPPrefix.off h p prog t.val j at no
      exact no.elim (fun ne => ne rfl) (fun ne => ne same)
    · have no := outside.1 (2 : Fin 3) (t,j,()) trivial
      change ("dg" : RegionName)≠"dg" ∨ address≠FPPrefix.off h p prog t.val j at no
      exact no.elim (fun ne => ne rfl) (fun ne => ne same)

 theorem source_run (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ)
    (vals : (io M kind h p prog scale).Inputs) (s : State α)
    (hin : (io M kind h p prog scale).InputBinding vals s) :
    ∃ final, runSchedule ((io M kind h p prog scale).schedule vals) s=some final ∧
      (∀ r∈FPTile.outputs kind, ∀ (t : Fin p.span) (j : Fin h.BK),
        final.mem r (FPPrefix.off h p prog t.val j)=
          FPLogical.evaluate M kind p.span scale
            (fun r n => logicalRead kind h p vals r n j) r t.val) ∧
      (io M kind h p prog scale).Frame vals s final := by
  let start := (launch M kind h p prog scale).initial s
  have hi : (io M kind h p prog scale).InputBinding vals start := hin
  obtain ⟨final,ran,values,frame⟩ := FPTile.complete M kind h p
    (phase_positive h p hp).2.1 prog scale start rfl rfl rfl
  refine ⟨final,?_,?_,?_⟩
  · simpa only [io,runSchedule_singleton] using ran
  · intro r member t j
    rw [values r member t j]
    apply FPLogical.tile_evaluate M kind h p (phase_positive h p hp).2.1
    intro r hr n hn
    exact input_reads M kind h p prog scale vals start hi r hr n hn j
  · exact raw_frame M kind h p prog scale vals s final frame

#axiomsClean source_run
end
end VeriTile.Bench.Repairs.DecayCumsum.DecayIO
