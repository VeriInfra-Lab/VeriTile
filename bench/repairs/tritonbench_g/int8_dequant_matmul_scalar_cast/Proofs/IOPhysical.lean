import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOInvocation
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOBinding
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteSchedule

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOPhysical
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

theorem initial_original (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (b : Buffers) (m : Int8DequantMatmul.Memory F H)
    (divisor : F) (p : PhysicalCommit s c) :
    (IO.invocation q s c hasBias divisor (IOPrograms.physicalProgram p)).initial
      (Quote.initialState q b m divisor p.pid.val p.split.val)=
        Quote.initialState q b m divisor p.pid.val p.split.val := by
  simp [IO.invocation,IOPrograms.physicalProgram,IntegerDotInvocation.initial,Invocation.initial,
    bindRegisters,Quote.initialState,State.setReg]
  funext dtype shape name
  by_cases same : name="divfactor" <;> simp [same]

/-- Every original physical commit extracts its descriptors from precisely
this shared complete-CTA invocation. No uniform CTA ordering is substituted
for the original independent per-address atomic schedule. -/
theorem run_original (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (b : Buffers) (m : Int8DequantMatmul.Memory F H)
    (divisor : F) (p : PhysicalCommit s c) :
    (IO.invocation q s c hasBias divisor (IOPrograms.physicalProgram p)).run
      (Quote.initialState q b m divisor p.pid.val p.split.val)=
        Quote.execute (Quote.floating q) q s.M s.N (Quote.source s c hasBias)
          (Quote.initialState q b m divisor p.pid.val p.split.val) := by
  rw [IOInvocation.run_eq,initial_original]

/-- The finite public inputs bind the original raw physical storage views,
including read-only aliases, element offsets and old typed-default reads. -/
theorem input_cells (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (b : Buffers) (m : Int8DequantMatmul.Memory F H)
    (divisor : F) (schedule : Schedule s c) (p : PhysicalCommit s c) :
    (IO.programIO q s c hasBias divisor schedule).InputCells (IOBinding.values q.ops s b m)
      (Quote.initialState q b m divisor p.pid.val p.split.val) :=
  IOBinding.source_cells q s c hasBias b m divisor p.pid.val p.split.val

#axiomsClean initial_original
#axiomsClean run_original
#axiomsClean input_cells
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOPhysical
