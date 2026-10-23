import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelState
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelBinding
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Real
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Schedule

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelResult
open VeriTile.Triton FP.Structural
open TypedExecution KernelBinding
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
variable {α : Type} [Inhabited α]

theorem checked_cell (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (context : State α) (i : Nat) (hi : i<rows*cols) :
    (finalState (rows*cols) cols (b.state context)).mem "output" i =
      .mk .int (signedValue (wordTarget rows cols i)) :=
  initialized_cell (b.state context) rows cols positive
    (b.input_cell context) (b.output_cell context) i hi

theorem checked_real (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (context : State α) (i : Nat) (hi : i<rows*cols) :
    (((finalState (rows*cols) cols (b.state context)).mem "output" i).read .int : ℝ) =
      realTarget rows cols i := by
  rw [checked_cell api memory b positive context i hi, Cell.read_same]
  exact int32_real_exact rows cols i

theorem checked_word (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (context : State α) (i : Nat) (hi : i<rows*cols) :
    BitVec.ofInt 32 (((finalState (rows*cols) cols (b.state context)).mem "output" i).read .int) =
      wordTarget rows cols i := by
  rw [checked_cell api memory b positive context i hi, Cell.read_same]
  simp [signedValue]

/-- Any within-store lane permutation gives the same actual physical words.
CTA barriers establish the inter-store order separately. -/
theorem physical_words (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (schedule : Schedule (rows*cols) cols) (i : Nat) (hi : i<rows*cols) :
    api.read (execute api 0 b.placement schedule.order memory) (b.output.dense.address i) =
      wordTarget rows cols i := by
  have columns : 0<cols := Nat.pos_of_mul_pos_left positive
  have h := execute_schedule_readback api 0 b.placement schedule columns memory i hi
  apply h.trans
  exact target_congr_inside 0 (rows*cols) cols columns
    (readInput api b.placement memory) arangeWord
    (fun j ↦ api.read memory (b.output.dense.address j)) (fun _ ↦ 0)
    (by intro j hj; exact b.inputArange j hj)
    (by intro j hj; exact b.outputZero j hj) i hi

theorem physical_bytes (api : API Word Store Byte) (codec : Codec api) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (schedule : Schedule (rows*cols) cols) (i byte : Nat)
    (hi : i<rows*cols) (hb : byte<4) :
    api.byte (execute api 0 b.placement schedule.order memory) (b.output.dense.address i+byte) =
      (wordTarget rows cols i).extractLsb' (8*byte) 8 := by
  rw [codec.byte _ _ _ hb, physical_words api memory b positive schedule i hi]

theorem typed_physical (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (context : State α) (schedule : Schedule (rows*cols) cols) (i : Nat) (hi : i<rows*cols) :
    (finalState (rows*cols) cols (b.state context)).mem "output" i =
      .mk .int (signedValue (api.read (execute api 0 b.placement schedule.order memory)
        (b.output.dense.address i))) := by
  rw [physical_words api memory b positive schedule i hi]
  exact checked_cell api memory b positive context i hi

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelResult

#guard_msgs (drop info) in
#auditModuleAxioms
