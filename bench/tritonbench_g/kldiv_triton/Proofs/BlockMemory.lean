import bench.tritonbench_g.kldiv_triton.Proofs.Statement

namespace VeriTile.Bench.Repairs.KldivTriton.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem scatter_regs fold_written fold_frame)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 3000000

structure Header {α : Type} (s : State α) (yBase tBase oBase B : Nat) : Prop where
  y : s.regs .ptr [] "y_ptr" = some (fun _ => (("y_ptr" : RegionName), yBase))
  t : s.regs .ptr [] "gt_ptr" = some (fun _ => (("gt_ptr" : RegionName), tBase))
  o : s.regs .ptr [] "loss_ptr" = some (fun _ => (("loss_ptr" : RegionName), oBase))
  base : s.regs .nat [B] "base_offsets" = some (fun j => j.1.val)

theorem initial_header {α : Type} (s : State α) (sy st so B : Nat) :
    Header (forwardInitial s sy st so B) (s.pids 0 * sy) (s.pids 0 * st) (s.pids 0 * so) B := by
  constructor <;> simp [forwardInitial]

theorem set_i_ready {α : Type} (m : Model α) (s : State α)
    (sy st so yBase tBase oBase N B current : Nat) (flag : Bool) (reduction : Nat)
    (h : Header s yBase tBase oBase B) :
    Ready (Invocation.mk m sy st so N B flag reduction yBase tBase oBase current
      (s.setReg "i" .int [] (fun _ => (current : Int)))) := by
  constructor <;> simp [h.y, h.t, h.o, h.base]

theorem middle_mem {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase N B current : Nat) (flag : Bool) :
    (f6 m s yBase tBase N B current flag).mem = s.mem := by
  cases flag <;> rfl

theorem result_header {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B current : Nat) (flag : Bool) (reduction : Nat)
    (h : Header s yBase tBase oBase B) :
    Header (forwardResult m s yBase tBase oBase N B current flag reduction)
      yBase tBase (oBase + if reduction = 0 then 0 else 1) B := by
  by_cases hr : reduction = 0 <;> cases flag
  all_goals constructor <;>
    simp [forwardResult, f6, f5, f4, f3, f2, f1, scatter, scatter_regs,
      hr, h.y, h.t, h.o, h.base]

theorem scatter_cell {α : Type} (s : State α) (region : RegionName) (base N B current : Nat)
    (values : TileIndex [B] → α) (target : TileIndex [B]) (active : current + target.1.val < N) :
    (scatter s region base N B current values).mem region (base + (current + target.1.val)) =
      .mk .real (values target) := by
  have unique : ∀ i : TileIndex [B], current + i.1.val < N →
      base + (current + i.1.val) = base + (current + target.1.val) → i = target := by
    intro i _ heq
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ := List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold scatter
  rw [scatter_mem, hall, List.foldl_append, List.foldl_cons]
  simp only [active, if_true]
  apply fold_written _ _ _ _ target unique
  simp [writeCell]

theorem scatter_preserves {α : Type} (s : State α) (region : RegionName) (base N B current : Nat)
    (values : TileIndex [B] → α) (r : RegionName) (a : Nat)
    (outside : r ≠ region ∨ ∀ j : TileIndex [B], current + j.1.val < N → a ≠ base + (current + j.1.val)) :
    (scatter s region base N B current values).mem r a = s.mem r a := by
  unfold scatter
  rw [scatter_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

def Footprint (oBase N B current reduction : Nat) (region : RegionName) (a : Nat) : Prop :=
  region = "loss_ptr" ∧ if reduction = 0 then
    ∃ j : Fin B, current + j.val < N ∧ a = oBase + (current + j.val)
  else a = oBase

theorem result_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B current : Nat) (flag : Bool) (reduction : Nat) (region : RegionName) (a : Nat)
    (outside : ¬Footprint oBase N B current reduction region a) :
    (forwardResult m s yBase tBase oBase N B current flag reduction).mem region a = s.mem region a := by
  by_cases hr : reduction = 0
  · rw [forwardResult, if_pos hr, scatter_preserves, middle_mem]
    by_cases same : region = "loss_ptr"
    · right
      intro j active equal
      apply outside
      exact ⟨same, by simpa [hr] using (show ∃ k : Fin B, current + k.val < N ∧ a = oBase + (current + k.val)
        from ⟨j.1, active, equal⟩)⟩
    · exact Or.inl same
  · have other : region ≠ "loss_ptr" ∨ a ≠ oBase := by
      by_contra h
      apply outside
      push_neg at h
      exact ⟨h.1, by simpa [hr] using h.2⟩
    simp only [forwardResult, hr, if_false, State.setReg_mem,
      State.write_other _ _ _ _ _ _ other, middle_mem]

theorem result_inputs {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B current : Nat) (flag : Bool) (reduction : Nat)
    (region : RegionName) (base : Nat) (different : region ≠ "loss_ptr") :
    input (forwardResult m s yBase tBase oBase N B current flag reduction) region base =
      input s region base := by
  funext col
  unfold input
  rw [result_frame m s yBase tBase oBase N B current flag reduction region (base + col)
    (fun h => different h.1)]

theorem result_element {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B current : Nat) (flag : Bool) (target : TileIndex [B])
    (active : current + target.1.val < N) :
    (forwardResult m s yBase tBase oBase N B current flag 0).mem "loss_ptr"
        (oBase + (current + target.1.val)) =
      .mk .real (loss m flag
        (m.alg.fp32Load (input s "y_ptr" yBase (current + target.1.val)))
        (m.alg.fp32Load (input s "gt_ptr" tBase (current + target.1.val)))) := by
  simp only [forwardResult, if_true]
  rw [scatter_cell _ _ _ _ _ _ _ target active]
  simp only [laneLoss, loaded, active, if_true]

theorem result_chunk {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B chunk reduction : Nat) (flag : Bool) (hr : reduction ≠ 0) :
    (forwardResult m s yBase tBase oBase N B (chunk * B) flag reduction).mem "loss_ptr" oBase =
      .mk .real (chunkSum m flag N B chunk (input s "y_ptr" yBase) (input s "gt_ptr" tBase)) := by
  simp [forwardResult, hr, chunkSum]

#axiomsClean result_header
#axiomsClean result_frame
#axiomsClean result_element
#axiomsClean result_chunk
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Tile
