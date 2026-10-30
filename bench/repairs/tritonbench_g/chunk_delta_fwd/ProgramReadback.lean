import bench.repairs.tritonbench_g.chunk_delta_fwd.Readback
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def Footprint (p : Parameters) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  (r="history" ∧ ∃ c, c < p.NT ∧ ∃ i : HIndex p, hActive p pid i ∧ o=hOffset p pid c i) ∨
  (r="corrected" ∧ ∃ c, c < p.NT ∧ ∃ b, b < innerCount p ∧ ∃ i : CIndex p,
    cActive p pid c b i ∧ o=cOffset p pid c b i) ∨
  (r="final" ∧ p.storeFinal=Bool.true ∧ ∃ i : HIndex p, hActive p pid i ∧ o=fOffset p pid i)

theorem program_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (r : RegionName) (o : Nat) (safe : ¬Footprint p pid r o) :
    programMemory M D p pid input m r o=m r o := by
  have hf : ∀ mem, writeFinal M p pid (states M D p pid input p.NT) mem r o=mem r o := by
    intro mem
    cases hp : p.storeFinal
    · simp [writeFinal,hp]
    · apply final_frame
      by_cases hr : r="final"
      · right
        intro i hi he
        exact safe (Or.inr (Or.inr ⟨hr,hp,i,hi,he⟩))
      · exact Or.inl hr
  unfold programMemory
  rw [hf]
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro c hc mem
  unfold writeOuter
  rw [inner_frame]
  · apply history_frame
    by_cases hr : r="history"
    · right
      intro i hi he
      exact safe (Or.inl ⟨hr,c,hc,i,hi,he⟩)
    · exact Or.inl hr
  · by_cases hr : r="corrected"
    · right
      intro b hb i hi he
      exact safe (Or.inr (Or.inl ⟨hr,c,hc,b,hb,i,hi,he⟩))
    · exact Or.inl hr

theorem program_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (c : Nat) (i : HIndex p) (hc : c < p.NT) (hi : hActive p pid i) :
    programMemory M D p pid input m "history" (hOffset p pid c i)=
      .mk .real (cast M (states M D p pid input c i)) := by
  unfold programMemory
  rw [final_frame _ _ _ _ _ _ _ (Or.inl (by decide))]
  apply foldRange_last _ (fun mem : Memory α => mem "history" (hOffset p pid c i)) p.NT c m _ hc
  · intro mem
    unfold writeOuter
    rw [inner_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
    exact history_read M p pid c _ mem i hc hi
  · intro d hd hdn mem
    unfold writeOuter
    rw [inner_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
    apply history_frame
    right
    intro j hj he
    have hcd := (Geometry.history_collision p pid pid c d i j hi hj hc hdn he).2.1
    omega

theorem program_corrected (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (c b : Nat) (i : CIndex p) (hc : c < p.NT)
    (hb : b < innerCount p) (hi : cActive p pid c b i)
    (hbase : time p c b i.1.val < (c+1)*p.BT) :
    programMemory M D p pid input m "corrected" (cOffset p pid c b i)=
      .mk .real (cast M (corrected M D p pid input c b (states M D p pid input c) i)) := by
  unfold programMemory
  rw [final_frame _ _ _ _ _ _ _ (Or.inl (by decide))]
  apply foldRange_last _ (fun mem : Memory α => mem "corrected" (cOffset p pid c b i)) p.NT c m _ hc
  · intro mem
    unfold writeOuter
    exact inner_read M D p pid input c b _ _ i hb hi
  · intro d hd _ mem
    unfold writeOuter
    rw [inner_frame]
    · exact history_frame _ _ _ _ _ _ _ _ (Or.inl (by decide))
    · right
      intro k _ j hj he
      have htime := (Geometry.corrected_collision p pid pid c d b k i j hi hj he).2.1
      have hmul := Nat.mul_le_mul_right p.BT hd
      change (c+1)*p.BT ≤ d*p.BT at hmul
      dsimp [time] at htime hbase
      omega

theorem program_final (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (i : HIndex p) (hp : p.storeFinal=Bool.true) (hi : hActive p pid i) :
    programMemory M D p pid input m "final" (fOffset p pid i)=
      .mk .real (cast M (states M D p pid input p.NT i)) :=
  final_read M p pid _ _ i hp hi

#axiomsClean program_frame
#axiomsClean program_history
#axiomsClean program_corrected
#axiomsClean program_final
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
