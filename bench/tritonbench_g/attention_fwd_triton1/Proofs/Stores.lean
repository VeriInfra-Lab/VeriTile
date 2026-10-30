import bench.tritonbench_g.attention_fwd_triton1.Proofs.StepFacts

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem main_injective (c : Shape) (head chunk : Nat) :
    Function.Injective (fun i : TileIndex [32,c.D] => mainAddress c head chunk i.1.val i.2.1.val) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ equal
  have digits := digit_injective c.D (chunk*32+i.val) j.val (chunk*32+i'.val) j'.val
    j.isLt j'.isLt (by simpa only [mainAddress,Nat.add_assoc,Nat.add_right_inj] using equal)
  have hi : i=i' := Fin.ext (by omega)
  have hj : j=j' := Fin.ext digits.2
  simp [hi,hj]

theorem history_injective (c : Shape) (head chunk : Nat) :
    Function.Injective (fun i : TileIndex [c.D,c.D] => historyAddress c head chunk i.1.val i.2.1.val) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ equal
  have digits := digit_injective c.D (chunk*c.D+i.val) j.val (chunk*c.D+i'.val) j'.val
    j.isLt j'.isLt (by simpa only [historyAddress,Nat.add_assoc,Nat.add_right_inj] using equal)
  have hi : i=i' := Fin.ext (by omega)
  have hj : j=j' := Fin.ext digits.2
  simp [hi,hj]

theorem scatter_frame (m n : Nat) (ptr : BlockPtr) (axes : List Nat)
    (values : Matrix α m n) (s : State (Term α)) (region : RegionName) (off : Nat)
    (miss : region≠ptr.region ∨ ∀ i : TileIndex [m,n],
      ptr.inBounds (TileShape.indexToList [m,n] i) axes=Bool.true →
      off≠ptr.address (TileShape.indexToList [m,n] i)) :
    (scatter m n ptr axes values s).mem region off=s.mem region off := by
  apply State.masked_scatter_frame
  rcases miss with hr|hi
  · exact Or.inl hr
  · exact Or.inr (fun i _ => hi i)

theorem finished_output [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α))
    (i : Fin 32) (j : Fin c.D) (active : chunk*32+i.val<c.T) :
    (finished c scale head chunk storeFlag ifcond carry s).mem "o"
      (mainAddress c head chunk i.val j.val)=.mk .real
        (TermBody.cast (output c (read s "q") (read s "k") (read s "v")
          scale head chunk carry ifcond (i,j,PUnit.unit))) := by
  unfold finished
  simpa only [scatter,TileShape.indexToList,main_bound,main_address,main_region,
    decide_eq_true_eq] using State.masked_scatter_readback (α := Term α) "o"
      (fun i : TileIndex [32,c.D] => mainAddress c head chunk i.1.val i.2.1.val)
      (fun i => Cell.mk .real (TermBody.cast (output c (read s "q") (read s "k") (read s "v")
        scale head chunk carry ifcond i)))
      (fun i => chunk*32+i.1.val<c.T) (updated c scale head chunk storeFlag ifcond carry s)
      (i,j,PUnit.unit) active (main_injective c head chunk)

@[simp] theorem history_region (c : Shape) (head chunk : Nat) :
    (historyPtr c head chunk).region="h" := rfl

theorem finished_history [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) (i j : Fin c.D) :
    (finished c scale head chunk Bool.true ifcond carry s).mem "h"
      (historyAddress c head chunk i.val j.val)=.mk .real (TermBody.cast (carry (i,j,PUnit.unit))) := by
  rw [finished,scatter_other _ _ _ _ _ _ "h" _ (by change ("h" : RegionName)≠"o"; decide),updated_memory]
  change (scatter c.D c.D (historyPtr c head chunk) [] (fun i => TermBody.cast (carry i))
    (pointers c head chunk s)).mem "h" (historyAddress c head chunk i.val j.val)=_
  simpa only [scatter,BlockPtr.inBounds_nil_checkedAxes,↓reduceIte,
    TileShape.indexToList,history_address,history_region] using State.scatter_readback
      (α := Term α) "h"
      (fun i : TileIndex [c.D,c.D] => historyAddress c head chunk i.1.val i.2.1.val)
      (fun i => Cell.mk .real (TermBody.cast (carry i))) (pointers c head chunk s)
      (i,j,PUnit.unit) (history_injective c head chunk)

theorem finished_output_before [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) (off : Nat)
    (before : off<head*c.headSize+chunk*32*c.D) :
    (finished c scale head chunk storeFlag ifcond carry s).mem "o" off=s.mem "o" off := by
  rw [finished,scatter_frame,updated_memory]
  · exact saved_input c head chunk storeFlag carry s "o" off (by decide)
  · right
    rintro ⟨i,j,⟨⟩⟩ _ equal
    simp only [TileShape.indexToList,main_address,mainAddress,Nat.add_mul] at equal
    omega

theorem finished_history_before [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) (off : Nat)
    (before : off<head*c.historyHead+chunk*c.D*c.D) :
    (finished c scale head chunk storeFlag ifcond carry s).mem "h" off=s.mem "h" off := by
  rw [finished,scatter_other _ _ _ _ _ _ "h" off (by change ("h" : RegionName)≠"o"; decide),updated_memory]
  cases storeFlag with
  | «false» => rfl
  | «true» =>
    apply scatter_frame
    right
    rintro ⟨i,j,⟨⟩⟩ _ equal
    simp only [TileShape.indexToList,history_address,historyAddress,Nat.add_mul] at equal
    omega

theorem finished_history_disabled [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) (off : Nat) :
    (finished c scale head chunk Bool.false ifcond carry s).mem "h" off=s.mem "h" off := by
  rw [finished,scatter_other _ _ _ _ _ _ "h" off (by change ("h" : RegionName)≠"o"; decide),updated_memory]
  rfl

theorem finished_frame [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α))
    (headBound : head<c.heads) (chunkBound : chunk<c.chunks) (r : RegionName) (off : Nat)
    (ho : r≠"o" ∨ c.mainSize≤off)
    (hh : storeFlag=Bool.false ∨ r≠"h" ∨ c.historySize≤off) :
    (finished c scale head chunk storeFlag ifcond carry s).mem r off=s.mem r off := by
  rw [finished,scatter_frame,updated_memory]
  · cases storeFlag with
    | «false» => rfl
    | «true» =>
      apply scatter_frame
      rcases hh with bad|different|bound
      · cases bad
      · exact Or.inl different
      · right
        rintro ⟨i,j,⟨⟩⟩ _ equal
        have inside := c.history_bound head chunk i.val j.val headBound chunkBound i.isLt j.isLt
        simp only [TileShape.indexToList,history_address] at equal
        change off=c.historyOffset head chunk i.val j.val at equal
        omega
  · rcases ho with different|bound
    · exact Or.inl different
    · right
      rintro ⟨i,j,⟨⟩⟩ active equal
      simp only [TileShape.indexToList,main_bound,decide_eq_true_eq] at active
      have inside := c.main_bound head (chunk*32+i.val) j.val headBound active j.isLt
      simp only [TileShape.indexToList,main_address] at equal
      change off=c.mainOffset head (chunk*32+i.val) j.val at equal
      omega

#axiomsClean finished_output
#axiomsClean finished_history
#axiomsClean finished_frame
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
