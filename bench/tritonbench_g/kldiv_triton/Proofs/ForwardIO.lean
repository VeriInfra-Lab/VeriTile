import bench.tritonbench_g.kldiv_triton.Proofs.Loop

namespace VeriTile.Bench.Repairs.KldivTriton.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds outputCount lt_rounds_iff)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

def RowFootprint (base N B reduction : Nat) (region : RegionName) (a : Nat) : Prop :=
  region = "loss_ptr" ∧ base ≤ a ∧ a < base + outputCount N B reduction

theorem tick_inputs {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B k : Nat) (flag : Bool) (reduction : Nat)
    (region : RegionName) (offset : Nat) (different : region ≠ "loss_ptr") :
    input (tick m s yBase tBase base N B k flag reduction) region offset = input s region offset := by
  unfold tick
  rw [result_inputs m _ _ _ _ _ _ _ _ _ region offset different]
  rfl

theorem iterate_inputs {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B count k : Nat) (flag : Bool) (reduction : Nat)
    (region : RegionName) (offset : Nat) (different : region ≠ "loss_ptr") :
    input (iterate m yBase tBase base N B flag reduction count k s) region offset = input s region offset := by
  induction count generalizing k s with
  | zero => rfl
  | succ count ih =>
    rw [iterate, ih, tick_inputs m s yBase tBase base N B k flag reduction region offset different]

theorem tick_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B k : Nat) (flag : Bool) (reduction : Nat)
    (hk : k < rounds N B) (region : RegionName) (a : Nat)
    (outside : ¬RowFootprint base N B reduction region a) :
    (tick m s yBase tBase base N B k flag reduction).mem region a = s.mem region a := by
  unfold tick
  rw [result_frame]
  · rfl
  · intro hit
    apply outside
    refine ⟨hit.1, ?_⟩
    by_cases hr : reduction = 0
    · obtain ⟨j, active, address⟩ := by simpa [Footprint, hr] using hit.2
      simp only [outputBase, hr, if_true, Nat.add_zero] at address
      simp only [outputCount, hr, if_true]
      omega
    · have address : a = base + k := by simpa [outputBase, hr] using hit.2
      simp only [outputCount, hr, if_false]
      omega

theorem iterate_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B count k : Nat) (flag : Bool) (reduction : Nat)
    (bounded : k + count ≤ rounds N B) (region : RegionName) (a : Nat)
    (outside : ¬RowFootprint base N B reduction region a) :
    (iterate m yBase tBase base N B flag reduction count k s).mem region a = s.mem region a := by
  induction count generalizing k s with
  | zero => rfl
  | succ count ih =>
    rw [iterate, ih _ (k + 1) (by omega)]
    exact tick_frame m s yBase tBase base N B k flag reduction (by omega) region a outside

theorem iterate_before {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B count k : Nat) (flag : Bool) (reduction : Nat) (a : Nat)
    (before : a < base + if reduction = 0 then k * B else k) :
    (iterate m yBase tBase base N B flag reduction count k s).mem "loss_ptr" a = s.mem "loss_ptr" a := by
  induction count generalizing k s with
  | zero => rfl
  | succ count ih =>
    have next : a < base + if reduction = 0 then (k + 1) * B else k + 1 := by
      by_cases hr : reduction = 0 <;> simp only [hr, if_true, if_false, Nat.add_mul, Nat.one_mul] at *
      · omega
      · omega
    rw [iterate, ih _ (k + 1) next]
    unfold tick
    rw [result_frame]
    · rfl
    · intro hit
      by_cases hr : reduction = 0
      · obtain ⟨j, _, address⟩ := by simpa [Footprint, hr] using hit.2
        simp only [outputBase, hr, if_true, Nat.add_zero] at address
        simp only [hr, if_true] at before
        omega
      · have address : a = base + k := by simpa [outputBase, hr] using hit.2
        simp only [hr, if_false] at before
        omega

theorem iterate_element {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B count k q : Nat) (flag : Bool) (lane : Fin B)
    (lower : k ≤ q) (upper : q < k + count) (active : q * B + lane.val < N) :
    (iterate m yBase tBase base N B flag 0 count k s).mem "loss_ptr" (base + (q * B + lane.val)) =
      .mk .real (loss m flag
        (m.alg.fp32Load (input s "y_ptr" yBase (q * B + lane.val)))
        (m.alg.fp32Load (input s "gt_ptr" tBase (q * B + lane.val)))) := by
  induction count generalizing k s with
  | zero => omega
  | succ count ih =>
    rw [iterate]
    by_cases same : q = k
    · subst q
      rw [iterate_before m _ yBase tBase base N B count (k + 1) flag 0 _ (by
        simp only [if_true, Nat.add_mul, Nat.one_mul]; have := lane.isLt; omega)]
      simpa only [tick, outputBase, if_true, Nat.add_zero, input, State.setReg_mem] using
        result_element m (s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int)))
          yBase tBase base N B (k*B) flag (lane, PUnit.unit) active
    · rw [ih _ (k + 1) (by omega) (by omega)]
      rw [tick_inputs m s yBase tBase base N B k flag 0 "y_ptr" yBase (by decide),
        tick_inputs m s yBase tBase base N B k flag 0 "gt_ptr" tBase (by decide)]

theorem iterate_chunk {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B count k q reduction : Nat) (flag : Bool) (hr : reduction ≠ 0)
    (lower : k ≤ q) (upper : q < k + count) :
    (iterate m yBase tBase base N B flag reduction count k s).mem "loss_ptr" (base + q) =
      .mk .real (chunkSum m flag N B q (input s "y_ptr" yBase) (input s "gt_ptr" tBase)) := by
  induction count generalizing k s with
  | zero => omega
  | succ count ih =>
    rw [iterate]
    by_cases same : q = k
    · subst q
      rw [iterate_before m _ yBase tBase base N B count (k + 1) flag reduction _ (by simp [hr])]
      simpa only [tick, outputBase, hr, if_false, input, State.setReg_mem] using
        result_chunk m (s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int)))
          yBase tBase (base+k) N B k reduction flag hr
    · rw [ih _ (k + 1) (by omega) (by omega)]
      rw [tick_inputs m s yBase tBase base N B k flag reduction "y_ptr" yBase (by decide),
        tick_inputs m s yBase tBase base N B k flag reduction "gt_ptr" tBase (by decide)]

theorem source_element {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (sy st so N B : Nat) (flag : Bool) (positive : 0 < B) (i : Nat) (active : i < N) :
    (sourceRun m s sy st so N B flag 0 positive).val.mem "loss_ptr" (s.pids 0 * so + i) =
      .mk .real (loss m flag
        (m.alg.fp32Load (input s "y_ptr" (s.pids 0 * sy) i))
        (m.alg.fp32Load (input s "gt_ptr" (s.pids 0 * st) i))) := by
  have cover : i / B * B + i % B = i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i B
  have q : i / B < rounds N B := (lt_rounds_iff N B (i/B) positive).mpr (by omega)
  have h := iterate_element m (forwardInitial s sy st so B) (s.pids 0*sy) (s.pids 0*st)
    (s.pids 0*so) N B (rounds N B) 0 (i/B) flag ⟨i%B,Nat.mod_lt i positive⟩
    (Nat.zero_le _) (by simpa only [Nat.zero_add] using q) (by simpa only [cover] using active)
  simpa only [sourceRun, cover, input, forwardInitial, State.setReg_mem] using h

theorem source_chunk {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0 < B)
    (hr : reduction ≠ 0) (k : Nat) (active : k < rounds N B) :
    (sourceRun m s sy st so N B flag reduction positive).val.mem "loss_ptr" (s.pids 0 * so + k) =
      .mk .real (chunkSum m flag N B k (input s "y_ptr" (s.pids 0*sy)) (input s "gt_ptr" (s.pids 0*st))) := by
  have h := iterate_chunk m (forwardInitial s sy st so B) (s.pids 0*sy) (s.pids 0*st)
    (s.pids 0*so) N B (rounds N B) 0 k reduction flag hr (by omega) (by omega)
  simpa only [sourceRun, input, forwardInitial, State.setReg_mem] using h

theorem source_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0 < B)
    (region : RegionName) (a : Nat) (outside : ¬RowFootprint (s.pids 0*so) N B reduction region a) :
    (sourceRun m s sy st so N B flag reduction positive).val.mem region a = s.mem region a := by
  exact iterate_frame m (forwardInitial s sy st so B) (s.pids 0*sy) (s.pids 0*st)
    (s.pids 0*so) N B (rounds N B) 0 flag reduction (by omega) region a outside

#axiomsClean source_element
#axiomsClean source_chunk
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Tile
