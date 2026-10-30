import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Backward
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section

theorem Shape.program_bound (c : Shape) (h q : Nat) (hh : h<c.heads) (hq : q<c.tiles) :
    h*c.tiles+q<c.programs := by
  calc
    h*c.tiles+q < h*c.tiles+c.tiles := Nat.add_lt_add_left hq _
    _ = (h+1)*c.tiles := by ring
    _ ≤ c.heads*c.tiles := Nat.mul_le_mul_right c.tiles (by omega)

namespace Source

def completeImage (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α) : Memory α :=
  fun r p t j =>
    if r=.output ∧ p<c.programs ∧ t<c.N ∧ j<1024 then
      Reference.forward ops c (inputsOf m) hasSeed p t j
    else if r=.dx ∧ p<c.programs ∧ t<c.N ∧ j<1024 then
      Reference.dx ops c (inputsOf m) p t j
    else if r=.tileGate ∧ p<c.programs ∧ t<c.N ∧ j=0 then
      Reference.tileGate ops c trees (inputsOf m) hasSeed p t
    else if r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 ∧ (c.N>0 ∨ hasSeed) then
      Reference.reverse ops c (inputsOf m) p c.N j
    else if r=.dgate ∧ p<c.heads ∧ t<c.N ∧ j=0 then
      Reference.dgate ops c trees (inputsOf m) hasSeed p t
    else m r p t j

theorem tileValue_forward (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α)
    (p t : Nat) (hp : p<c.programs) (ht : t+1<c.N) :
    tileValue ops c trees (forwardImage ops c hasSeed m c.programs) p t =
      Reference.tileGate ops c trees (inputsOf m) hasSeed p t := by
  unfold tileValue Reference.tileGate
  rw [if_pos ht,forwardImage_inputs]
  congr 1
  funext j
  simp [forwardImage,hp,show t<c.N by omega,j.isLt]

theorem completedScratch (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α)
    (p t : Nat) (hp : p<c.programs) (ht : t<c.N) :
    zeroTail ops c (backwardImage ops c trees (forwardImage ops c hasSeed m c.programs) c.programs)
      .tileGate p t 0 = Reference.tileGate ops c trees (inputsOf m) hasSeed p t := by
  by_cases hlast : t=c.N-1
  · simp [zeroTail,hp,hlast,show c.N>0 by omega,Reference.tileGate,show ¬c.N-1+1<c.N by omega]
  · simpa [zeroTail,hlast,backwardImage,hp,show t+1<c.N by omega] using
      (tileValue_forward ops c trees hasSeed m p t hp (by omega))

theorem run_complete (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α) :
    run ops c trees hasSeed m=completeImage ops c trees hasSeed m := by
  unfold run
  by_cases hz : c.N=0
  · rw [if_pos hz]
    funext r p t j
    simp only [emptyBackward,completeImage,hz,Nat.not_lt_zero,false_and,and_false,↓reduceIte,
      Reference.reverse,Nat.lt_irrefl,or_false,false_or]
    split_ifs <;> simp_all
  · rw [if_neg hz]
    simp only [forwardGrid_complete,backwardGrid_complete]
    funext r p t j
    by_cases hgate : r=.dgate ∧ p<c.heads ∧ t<c.N ∧ j=0
    · obtain ⟨rfl,hp,ht,rfl⟩ := hgate
      simp only [hostSum,completeImage,reduceCtorEq,false_and,true_and,and_true,hp,ht,↓reduceIte]
      unfold Reference.dgate
      congr 1
      funext q
      exact completedScratch ops c trees hasSeed m _ t (c.program_bound p q hp q.isLt) ht
    · simp only [hostSum,if_neg hgate]
      have hn : c.N>0 := by omega
      have ha : (t+1<c.N) ↔ (t<c.N ∧ t≠c.N-1) := by omega
      cases r with
      | dx =>
        by_cases hp : p<c.programs <;> by_cases ht : t<c.N <;>
          by_cases hj : j<1024 <;> by_cases hl : t=c.N-1 <;>
          simp_all [zeroTail,backwardImage,completeImage,Reference.dx,forwardImage,forwardImage_inputs] <;>
          split_ifs <;> (try simp_all) <;> omega
      | tileGate =>
        by_cases hp : p<c.programs
        · by_cases ht : t<c.N
          · by_cases hj : j=0
            · subst j
              simpa [completeImage,hp,ht] using completedScratch ops c trees hasSeed m p t hp ht
            · simp [zeroTail,backwardImage,completeImage,forwardImage,hj]
          · have ht1 : ¬t+1<c.N := by omega
            have hl : t≠c.N-1 := by omega
            simp [zeroTail,backwardImage,completeImage,forwardImage,ht,ht1,hl]
        · simp [zeroTail,backwardImage,completeImage,forwardImage,hp]
      | output | dseed | x | gate | seed | incoming | dgate | caller =>
        simp_all [zeroTail,backwardImage,completeImage,forwardImage,forwardImage_inputs]

/-- Exact write footprint of the combined call, including internal scratch
and DL even when the optional seed gradient is not returned. -/
def writes (c : Shape) (seed : Bool) (r : Region) (p t j : Nat) : Prop :=
  (r=.output ∧ p<c.programs ∧ t<c.N ∧ j<1024) ∨
  (r=.dx ∧ p<c.programs ∧ t<c.N ∧ j<1024) ∨
  (r=.tileGate ∧ p<c.programs ∧ t<c.N ∧ j=0) ∨
  (r=.dseed ∧ p<c.programs ∧ t=0 ∧ j<1024 ∧ (c.N>0 ∨ seed)) ∨
  (r=.dgate ∧ p<c.heads ∧ t<c.N ∧ j=0)

theorem run_frame (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory α)
    (r : Region) (p t j : Nat) (outside : ¬writes c seed r p t j) :
    run ops c trees seed m r p t j=m r p t j := by
  rw [run_complete]
  simp only [writes,not_or] at outside
  simp only [completeImage,if_neg outside.1,if_neg outside.2.1,if_neg outside.2.2.1,
    if_neg outside.2.2.2.1,if_neg outside.2.2.2.2]

/-- Complete caller frame, including arbitrary untouched allocation gaps and
all input/alias observations. No initial-zero assumption is used anywhere. -/
theorem caller_frame (ops : Ops α) (c : Shape) (trees : Trees c) (hasSeed : Bool) (m : Memory α)
    (r : Region) (p t j : Nat)
    (hr : r=.x ∨ r=.gate ∨ r=.seed ∨ r=.incoming ∨ r=.caller) :
    run ops c trees hasSeed m r p t j=m r p t j := by
  rw [run_complete]
  rcases hr with rfl|rfl|rfl|rfl|rfl <;> rfl

#axiomsClean run_complete
#axiomsClean caller_frame
#axiomsClean run_frame
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
