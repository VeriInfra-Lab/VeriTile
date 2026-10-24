import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSourceRun
import bench.tritonbench_g.isfinite_kernel.Proofs.FP

/-! The independently quoted candidate also transports by its actual source
body. Regional source equality alone is not used to infer flat execution. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

def candidateLoopTile : Op .nat [] :=
  .add .nat .nil (.ref .nat [] "pid")
    (.mul .nat .nil (.ref .nat [] "num_ctas") (.ref .nat [] "j"))

theorem candidate_loop32_exact (p : Parametric.Parameters) :
    Parametric.candidateLoop32 p = sourceBody p Bool.false candidateLoopTile := rfl
theorem candidate_loop64_exact (p : Parametric.Parameters) :
    Parametric.candidateLoop64 p = sourceBody p Bool.true candidateLoopTile := rfl

theorem candidate32_body_exact (p : Parametric.Parameters) :
    (Parametric.candidate32 p).surfaceBody =
      sourceProgramBody p Bool.false candidateLoopTile := by
  rw [Parametric.candidate32_split, source_one32_exact, candidate_loop32_exact]
  rfl
theorem candidate64_body_exact (p : Parametric.Parameters) :
    (Parametric.candidate64 p).surfaceBody =
      sourceProgramBody p Bool.true candidateLoopTile := by
  rw [Parametric.candidate64_split, source_one64_exact, candidate_loop64_exact]
  rfl

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

theorem candidate_loop_tile_same (s : State α) :
    OpSame A M background flatUndef s none candidateLoopTile := by
  apply op_add A M background flatUndef s none
  · exact op_ref_data A M background flatUndef s none .nat [] "pid" (by decide) (by decide)
  · apply op_mul A M background flatUndef s none
    · exact op_ref_data A M background flatUndef s none .nat [] "num_ctas" (by decide) (by decide)
    · exact op_ref_data A M background flatUndef s none .nat [] "j" (by decide) (by decide)

theorem candidate32_flat_exec (hd : A.Disjoint) (p : Parametric.Parameters)
    (s t : State α) (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (ht : FP.Structural.exec M (Parametric.candidate32 p) s = some t) :
    FP.Structural.exec M (flattenComputeKernel A (Parametric.candidate32 p))
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t) := by
  apply exec_of_transport A M background flatUndef _ s t ?_ ht
  rw [candidate32_body_exact]
  exact source_program_transport A M background flatUndef hd p Bool.false candidateLoopTile
    (candidate_loop_tile_same A M background flatUndef) s hu hinput houtput hin hout

theorem candidate64_flat_exec (hd : A.Disjoint) (p : Parametric.Parameters)
    (s t : State α) (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (ht : FP.Structural.exec M (Parametric.candidate64 p) s = some t) :
    FP.Structural.exec M (flattenComputeKernel A (Parametric.candidate64 p))
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t) := by
  apply exec_of_transport A M background flatUndef _ s t ?_ ht
  rw [candidate64_body_exact]
  exact source_program_transport A M background flatUndef hd p Bool.true candidateLoopTile
    (candidate_loop_tile_same A M background flatUndef) s hu hinput houtput hin hout

#axiomsClean candidate_loop32_exact
#axiomsClean candidate_loop64_exact
#axiomsClean candidate32_body_exact
#axiomsClean candidate64_body_exact
#axiomsClean candidate_loop_tile_same
#axiomsClean candidate32_flat_exec
#axiomsClean candidate64_flat_exec
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
