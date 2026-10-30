import bench.tritonbench_g.context_attn_llama.FPReferenceProgram

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
open VeriTile.Triton FP.Structural FPSuccess
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Reference host algorithm: enumerate query tiles and scatter the independent
online recurrence's values to their masked output coordinates. -/
def tile (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (idx : GridIndex c.grid) (s : State α) : State α :=
  let launch := atIndex s idx
  scatter c.layout c.regions launch launch
    (output c.layout M dots (Host.scale c.layout.D) (inputs c.layout c.regions launch))

def grid (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M) :
    List (GridIndex c.grid) → State α → State α
  | [],s => s
  | idx::rest,s => grid c M dots rest (tile c M dots idx s)

theorem tile_mem_congr (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (idx : GridIndex c.grid) (s t : State α) (memory : s.mem=t.mem) :
    (tile c M dots idx s).mem=(tile c M dots idx t).mem := by
  have inp : inputs c.layout c.regions (atIndex s idx)=inputs c.layout c.regions (atIndex t idx) := by
    simp only [inputs,atIndex,memory]
  dsimp only [tile]
  rw [inp]
  classical
  unfold scatter
  simp only [FPSuccess.active,FPSuccess.offset,FPSuccess.start,FPSuccess.length,FPSuccess.row,atIndex,memory]
  apply masked_fold_mem_congr
  rfl

theorem grid_mem_congr (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s t : State α) (memory : s.mem=t.mem) :
    (grid c M dots order s).mem=(grid c M dots order t).mem := by
  induction order generalizing s t with
  | nil => exact memory
  | cons idx rest ih =>
    exact ih _ _ (tile_mem_congr c M dots idx s t memory)

/-- Full source grid computes the independently defined recurrence grid. -/
theorem whole (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s : State α) :
    ∃ t,runGrid c M order s=some t ∧ t.mem=(grid c M dots order s).mem := by
  have bn : 0 < c.layout.BN := by rw [c.blockN]; exact Host.bm_positive c
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,he,hm⟩ := program c.layout c.regions (Host.scale c.layout.D) M dots bn (atIndex s idx)
    obtain ⟨final,hfinal,hfm⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [runGrid,he,Option.bind_some]
      exact hfinal
    · exact hfm.trans (grid_mem_congr c M dots rest mid (tile c M dots idx s) hm)

#specNonCircular grid avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,kernel,parts,runGrid]
#axiomsClean whole
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
