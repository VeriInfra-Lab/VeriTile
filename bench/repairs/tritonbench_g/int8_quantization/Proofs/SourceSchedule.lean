import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceNative

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Schedule
noncomputable section
set_option maxHeartbeats 5000000

def limit : Nat := 2147483647

structure Chunk where
  base : Nat
  count : Nat
  deriving DecidableEq

structure Program where
  base : Nat
  pid : Nat
  deriving DecidableEq

def Program.index (p : Program) : Nat := p.pid+p.base
def Chunk.programs (c : Chunk) : List Program := (List.range c.count).map (fun pid=>⟨c.base,pid⟩)

/-- The actual host range/min loop. Except after the last chunk (where no
further launch occurs), count=limit and the next BASE is BASE+limit. -/
def chunks (base remaining : Nat) : List Chunk :=
  if empty : remaining=0 then []
  else let count:=min remaining limit
    ⟨base,count⟩::chunks (base+count) (remaining-count)
termination_by remaining
decreasing_by
  have positive : 0< min remaining limit := by unfold limit;omega
  exact Nat.sub_lt (by omega) positive

def programs (base remaining : Nat) : List Program := (chunks base remaining).flatMap Chunk.programs

theorem chunk_indices (c : Chunk) : c.programs.map Program.index=List.range' c.base c.count := by
  simp [Chunk.programs,Program.index,List.map_map,List.range'_eq_map_range,Nat.add_comm]

theorem indices (base remaining : Nat) :
    (programs base remaining).map Program.index=List.range' base remaining := by
  induction remaining using Nat.strong_induction_on generalizing base with
  | h remaining ih=>
    by_cases empty : remaining=0
    · subst remaining;simp [programs,chunks]
    · have smaller : remaining-min remaining limit< remaining := Nat.sub_lt (by omega) (by unfold limit;omega)
      rw [programs,chunks]
      simp only [dif_neg empty,List.flatMap_cons,List.map_append,chunk_indices]
      rw [show ((chunks (base+min remaining limit) (remaining-min remaining limit)).flatMap Chunk.programs).map Program.index=
        List.range' (base+min remaining limit) (remaining-min remaining limit) from ih _ smaller _]
      rw [List.range'_append_1]
      congr 1
      omega

theorem chunk_bounds (base remaining : Nat) (c : Chunk) (member : c∈chunks base remaining) :
    base≤ c.base ∧ 0< c.count ∧ c.count≤ limit ∧ c.base+c.count≤ base+remaining := by
  induction remaining using Nat.strong_induction_on generalizing base with
  | h remaining ih=>
    by_cases empty : remaining=0
    · subst remaining;simp [chunks] at member
    · rw [chunks] at member
      simp only [dif_neg empty,List.mem_cons] at member
      rcases member with head|tail
      · subst c;simp only;unfold limit;omega
      · have smaller : remaining-min remaining limit< remaining := Nat.sub_lt (by omega) (by unfold limit;omega)
        obtain ⟨lo,pos,hi,upper⟩:=ih _ smaller _ tail
        exact ⟨by omega,pos,hi,by omega⟩

theorem program_bounds (base remaining : Nat) (w : Program) (member : w∈programs base remaining) :
    w.pid<2^31 ∧ base≤ w.index ∧ w.index< base+remaining := by
  obtain ⟨c,hc,hw⟩:=List.mem_flatMap.mp member
  obtain ⟨pid,hpid,eq⟩:=List.mem_map.mp hw
  cases eq
  have hp:=List.mem_range.mp hpid
  obtain ⟨lo,pos,hi,upper⟩:=chunk_bounds base remaining c hc
  dsimp [Program.index]
  unfold limit at hi
  omega

theorem complete (remaining g : Nat) (bound : g< remaining) :
    ∃ w∈programs 0 remaining,w.index=g := by
  have member : g∈(programs 0 remaining).map Program.index := by
    rw [indices]
    simpa [List.mem_range'] using bound
  exact List.mem_map.mp member

theorem unique (base remaining : Nat) : ((programs base remaining).map Program.index).Nodup := by
  rw [indices]
  exact List.nodup_range'

#axiomsClean indices
#axiomsClean program_bounds
#axiomsClean complete
#axiomsClean unique
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Schedule
