import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.IO

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IOExecution
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem values_congr (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s t : State α)
    (memory : s.mem=t.mem) (pids : s.pids=t.pids) : outputValues M R p s=outputValues M R p t := by
  have hp : ∀ n,products M p s n=products M p t n := by
    intro n
    funext i
    simp only [products,probability,value,gather,probOffset,gatherOffset,valueOffset,count,
      startIndex,requestIndex,sequenceLength,windowStart,memory,pids]
  have hu : update M R p s=update M R p t := by
    funext n a i
    simp only [update,hp]
  funext i
  simp only [outputValues,accumulated,hu,count,memory,pids]

theorem reference_mem (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s t : State α)
    (memory : s.mem=t.mem) (pids : s.pids=t.pids) :
    (reference M R p s).mem=(reference M R p t).mem := by
  simp only [reference,fold_write_memory,values_congr M R p s t memory pids,outputOffset,pids,memory]

theorem launch_complete (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (pid : Grid.PID) (s : State α) :
    ∃ t,(IO.launch M R p pid).run s=some t ∧
      t.mem=(reference M R p (Grid.state s.mem pid)).mem := by
  obtain ⟨t,ran,mem⟩ := complete_run M R p ((IO.launch M R p pid).initial s)
  refine ⟨t,ran,mem.trans ?_⟩
  apply reference_mem
  · rfl
  · rfl

theorem complete (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (s : State α) :
    ∃ t,runSchedule (work.map (IO.launch M R p)) s=some t ∧ t.mem=Grid.graph M R p work s.mem := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨middle,ran,mem⟩ := launch_complete M R p pid s
    obtain ⟨final,tail,eq⟩ := ih middle
    refine ⟨final,?_,?_⟩
    · simp only [List.map_cons,runSchedule,ran,Option.bind_some,tail]
    · simpa only [Grid.graph,mem] using eq

theorem launch_agree (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (pid : Grid.PID) (s t : State α) (separate : Grid.Separate p)
    (ran : (IO.launch M R p pid).run s=some t) : Grid.Agree p t.mem s.mem := by
  obtain ⟨u,hu,mem⟩ := launch_complete M R p pid s
  rw [ran] at hu
  injection hu with eq
  subst u
  rw [mem]
  exact Grid.block_agrees M R p pid s.mem separate

#axiomsClean complete
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IOExecution
