import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedStoreLists

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Finite store concatenation; all scalar payloads have already been derived
from the complete shared source execution. -/
def timeWrites (stop : Nat) (one : Nat → List (Write α)) (t : Nat) : List (Write α) :=
  if t<stop then one t++timeWrites stop one (t+1) else []
termination_by stop-t
decreasing_by omega

theorem covers_timeWrites (stop : Nat) (one : Nat → List (Write α)) (t : Nat)
    (r : RegionName) (i : Nat) :
    covers (timeWrites stop one t) r i ↔ ∃ n,t ≤ n ∧ n<stop ∧ covers (one n) r i := by
  by_cases ht : t<stop
  · rw [timeWrites,if_pos ht,covers_append,covers_timeWrites]
    constructor
    · rintro (h | ⟨n,hn,hn',h⟩)
      · exact ⟨t,le_rfl,ht,h⟩
      · exact ⟨n,by omega,hn',h⟩
    · rintro ⟨n,hn,hn',h⟩
      by_cases he : n=t
      · exact Or.inl (he ▸ h)
      · exact Or.inr ⟨n,by omega,hn',h⟩
  · rw [timeWrites,if_neg ht]
    simp only [covers,false_iff]
    rintro ⟨n,hn,hn',_⟩
    omega
termination_by stop-t
decreasing_by omega

theorem mem_timeWrites (stop : Nat) (one : Nat → List (Write α)) (t : Nat) (w : Write α) :
    w∈timeWrites stop one t ↔ ∃ n,t ≤ n ∧ n<stop ∧ w∈one n := by
  by_cases ht : t<stop
  · rw [timeWrites,if_pos ht,List.mem_append,mem_timeWrites]
    constructor
    · rintro (h | ⟨n,hn,hn',h⟩)
      · exact ⟨t,le_rfl,ht,h⟩
      · exact ⟨n,by omega,hn',h⟩
    · rintro ⟨n,hn,hn',h⟩
      by_cases he : n=t
      · exact Or.inl (he ▸ h)
      · exact Or.inr ⟨n,by omega,hn',h⟩
  · rw [timeWrites,if_neg ht]
    simp only [List.not_mem_nil,false_iff]
    rintro ⟨n,hn,hn',_⟩
    omega
termination_by stop-t
decreasing_by omega

theorem forwardTail_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) :
    (forwardTail a s b plans initial seed pid t u).mem=
      (writes (timeWrites s.T (outputWrites a s b plans initial seed pid) t) u).mem := by
  by_cases active : t<s.T
  · rw [forwardTail,if_pos active,timeWrites,if_pos active,writes_append,forwardTail_mem]
    apply writes_mem_congr
    exact forwardBodyResult_mem a s b plans initial u seed pid t
  · rw [forwardTail,if_neg active,timeWrites,if_neg active,writes_nil]
termination_by s.T-t
decreasing_by omega

theorem backwardFirstTail_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) :
    (backwardFirstTail a s b plans initial seed pid t u).mem=
      (writes (timeWrites s.T (queryWrites a s b plans initial seed pid) t) u).mem := by
  by_cases active : t<s.T
  · rw [backwardFirstTail,if_pos active,timeWrites,if_pos active,writes_append,backwardFirstTail_mem]
    apply writes_mem_congr
    exact backwardFirstBodyResult_mem a s b plans initial u seed pid t
  · rw [backwardFirstTail,if_neg active,timeWrites,if_neg active,writes_nil]
termination_by s.T-t
decreasing_by omega

theorem reverseTail_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId)
    (r : Nat) (u : State (Staged.Node α)) :
    (reverseTail a s b plans initial final pid r u).mem=
      (writes (timeWrites s.T (reverseWrites a s b plans initial final pid) r) u).mem := by
  by_cases active : r<s.T
  · rw [reverseTail,if_pos active,timeWrites,if_pos active,writes_append,reverseTail_mem]
    apply writes_mem_congr
    exact reverseBodyResult_mem a s b plans initial u final pid r
  · rw [reverseTail,if_neg active,timeWrites,if_neg active,writes_nil]
termination_by s.T-r
decreasing_by omega

def forwardStores [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) : List (Write (Staged.Node α)) :=
  timeWrites s.T (outputWrites a s b plans initial seed pid) 0++finalWrites a s b initial seed final pid

def backwardStores [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) : List (Write (Staged.Node α)) :=
  timeWrites s.T (queryWrites a s b plans initial seed pid) 0++
    timeWrites s.T (reverseWrites a s b plans initial final pid) 0

theorem forward_full_stores [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (seed final : Bool) (initial : State (Staged.Node α))
    (hBK : 0<s.BK) (covered : plans.output.Covers s.BK)
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK))) :
    ∃ result, FP.Structural.exec (Staged.algebra a plans)
      (Program.forward (layout s b) forwardRegions seed final) initial=some result ∧
      result.mem=(writes (forwardStores a s b plans initial seed final (launch initial)) initial).mem := by
  obtain ⟨entry,_,hmem,he⟩ := forward_full_values a s b plans seed final initial hBK covered hscale
  refine ⟨_,he,?_⟩
  rw [forwardFinalResult_mem,forwardStores,writes_append]
  apply writes_mem_congr
  rw [forwardTail_mem]
  exact writes_mem_congr _ _ _ hmem

theorem backward_full_stores [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (seed final : Bool) (initial : State (Staged.Node α))
    (hBK : 0<s.BK) (hBV : 0<s.BV) (coveredQ : plans.dq.Covers s.BV)
    (coveredK : plans.dk.Covers s.BV) (coveredV : plans.dv.Covers s.BK)
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK))) :
    ∃ result, FP.Structural.exec (Staged.algebra a plans)
      (Program.backward (layout s b) backwardRegions seed final) initial=some result ∧
      result.mem=(writes (backwardStores a s b plans initial seed final (launch initial)) initial).mem := by
  obtain ⟨entry,middle,_,hmem,_,hm,he⟩ := backward_full_values a s b plans seed final initial
    hBK hBV coveredQ coveredK coveredV hscale
  refine ⟨_,he,?_⟩
  rw [reverseTail_mem,backwardStores,writes_append]
  apply writes_mem_congr
  rw [hm,backwardFirstTail_mem]
  exact writes_mem_congr _ _ _ hmem

#axiomsClean forward_full_stores
#axiomsClean backward_full_stores
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
