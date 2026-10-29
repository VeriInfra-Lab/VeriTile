import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.KernelExecution
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteSchedule

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.KernelInterface
noncomputable section
set_option maxHeartbeats 3000000

/-- The full split schedule on raw buffers, including the required initial
FP16 zero cells for split-K and the independent native word recurrence. -/
theorem output_word (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (fresh : b.Fresh) (selected : Selection s c) (divisor : F) (bias : Bool)
    (q : Schedule s c) (initial : Memory F H) (zero : Zeroed a s c b initial)
    (cell : MatrixCell s) :
    gridRun a s c b selected divisor bias q.events initial (outputAddress s b cell)=
      .inr (.inr (wordReference a s c b selected divisor bias q initial cell)) := by
  have nonempty : projection cell q.events≠[] := by
    intro h
    have len := (q.complete cell).length_eq
    rw [h] at len
    simp only [List.length_nil,List.length_finRange] at len
    have positive := c.positiveS
    omega
  obtain ⟨value,hv⟩ := gridRun_typed a s c b selected divisor bias q.events initial cell (Or.inr nonempty)
  have result := gridRun_cell a s c b fresh selected divisor bias q.events initial initial (fun _ _ => rfl) cell
  have word : read16 a (gridRun a s c b selected divisor bias q.events initial) (outputAddress s b cell)=
      wordReference a s c b selected divisor bias q initial cell := by
    rw [result]
    by_cases single : c.S=1
    · exact cellFold_seed a c _ _ (q.complete cell) single _ _
    · rw [zeroed_read zero single cell]
      rfl
  simp only [read16,hv] at word
  rw [hv,word]

#axiomsClean output_word
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.KernelInterface
