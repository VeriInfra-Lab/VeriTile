import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def scheduledMemory (h : Input) (values : Inputs A h) (jobs : List Nat)
    (initial : State Word) : RegionName → Nat → Cell Word :=
  fun r offset => if live : r=h.output ∧ offset<h.shape.count*2 ∧ offset/1024∈jobs then
    .mk .bf16 (embed ⟨Bits.expectedBits (nibble values ⟨offset,live.2.1⟩)⟩)
  else initial.mem r offset

theorem schedule_result (rest : Algebra Word) (h : Input) (values : Inputs A h)
    (domain : ByteDomain h values) (jobs : List Nat) (s : State Word)
    (fresh : h.input≠h.output) (reads : Reads h values s) :
    ∃ final,HostSchedule.runSchedule rest h jobs s=some final ∧
      final.mem=scheduledMemory h values jobs s := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by funext r offset;simp [scheduledMemory]⟩
  | cons pid jobs ih =>
    let next := (execution rest (Host.parameters h) (programState h pid s)).val
    have run : exec (backend rest) (source (Host.parameters h)) (programState h pid s)=some next :=
      (execution rest (Host.parameters h) (programState h pid s)).property
    have nextReads : Reads h values next := by
      intro i
      have mem := program_memory rest h values domain s reads pid h.input i.val
      simp only [fresh,false_and,dite_false] at mem
      change (next.mem h.input i.val).read .nat=_
      rw [mem]
      exact reads i
    obtain ⟨final,ran,memory⟩ := ih next nextReads
    refine ⟨final,?_,?_⟩
    · simp only [HostSchedule.runSchedule,run,Option.bind_some]
      exact ran
    · rw [memory]
      funext r offset
      unfold scheduledMemory
      have mem := program_memory rest h values domain s reads pid r offset
      change next.mem r offset=_ at mem
      rw [mem]
      by_cases hr : r=h.output <;> by_cases hn : offset<h.shape.count*2 <;>
        by_cases hp : offset/1024=pid <;> by_cases hj : offset/1024∈jobs <;>
        simp [hr,hn,hp,hj]

theorem complete_result (rest : Algebra Word) (h : Input) (values : Inputs A h)
    (domain : ByteDomain h values) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (s : State Word) (fresh : h.input≠h.output) (reads : Reads h values s)
    (gridSafe : grid h≤2147483647) (order : jobs.Perm (List.range (grid h))) :
    ∃ final,Launch.run rest h jobs s=some final ∧ final.mem=targetMemory h values s := by
  obtain ⟨previous,ran,memory⟩ := schedule_result rest h values domain jobs s fresh reads
  obtain ⟨final,actual,same⟩ :=
    Launch.complete_transfer rest h storage jobs s previous gridSafe order ran
  refine ⟨final,actual,?_⟩
  rw [same,memory]
  funext r offset
  have cover := grid_covers h
  by_cases hr : r=h.output <;> by_cases hn : offset<h.shape.count*2
  · have member : offset/1024∈jobs := order.mem_iff.mpr (List.mem_range.mpr (by omega))
    simp [scheduledMemory,targetMemory,hr,hn,member]
  · simp [scheduledMemory,targetMemory,hr,hn]
  · simp [scheduledMemory,targetMemory,hr,hn]
  · simp [scheduledMemory,targetMemory,hr,hn]

theorem invocation_run (rest : Algebra Word) (h : Input) (pid : Nat) (s : State Word) :
    (invocation rest h pid).run s=Launch.program rest h pid s := rfl

theorem schedule_run (rest : Algebra Word) (h : Input) (jobs : List Nat) (s : State Word) :
    FP.Structural.runSchedule (jobs.map (invocation rest h)) s=Launch.run rest h jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,FP.Structural.runSchedule,invocation_run,Launch.run]
    congr 1
    funext next
    exact ih next

#axiomsClean complete_result
#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
