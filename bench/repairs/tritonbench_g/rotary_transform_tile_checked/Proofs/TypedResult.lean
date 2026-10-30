import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedExecution

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs fold_writes_mem)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The complete quotation has exactly the independent matrix-load/store
memory result. Register contents are internal to a source program. -/
theorem memory_result {α : Type} [Inhabited α] (p : Ops α) (s : Shape)
    (cumulative64 offsets64 : Bool) (initial : State α) :
    (FP.Structural.exec (KernelAlgebra.backend p)
      (kernel s cumulative64 offsets64) initial).map State.mem =
      some (ProgramModel.result s p initial).mem := by
  have hzero : (0.0 : ℝ)=0 := by norm_num
  have hone : (1.0 : ℝ)=1 := by norm_num
  rw [execution_result,Option.map_some]
  apply congrArg some
  by_cases early : ProgramModel.sequenceLength s initial ≤
      (((initial.pids 0*s.blockM) : Nat) : Int)
  · simp only [resultState,not_lt_of_ge early,if_false,setup_mem,
      ProgramModel.result,early,if_true]
  have active := lt_of_not_ge early
  simp only [resultState,active,if_true,ProgramModel.result,early,if_false]
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true <;> by_cases hc : s.conjugate = Bool.true
  all_goals
    simp (config := { maxSteps := 100000 }) [finish,loads,indices,setup,hv,ht,hi,hc,
      numeric,bop,Region.cast,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.insertAxis,TileShape.dropInsertedIndex,
      Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,integralMod,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,↓fold_writes_mem,hzero,hone,
      ProgramModel.scatter,ProgramModel.sequenceLength,
      ProgramModel.interleavedValue,ProgramModel.splitValue,ProgramModel.cosine,
      ProgramModel.sine,ProgramModel.hasCoefficient,ProgramModel.active,
      ProgramModel.dataOffset,ProgramModel.dataBase,ProgramModel.row,
      ProgramModel.coefficientOffset,ProgramModel.shift,ProgramModel.swapped,
      KernelAlgebra.backend,KernelAlgebra.regionName,Shape.output,outputRegion,
      Int.natCast_mul,Int.natCast_add]

  all_goals cases inplace : s.inplace <;> simp [inplace]

#axiomsClean memory_result
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
