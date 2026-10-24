import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatAssign
import bench.tritonbench_g.isfinite_kernel.Proofs.Header

/-! The real source prefix, from arbitrary initial typed state. Both source
assignments are retained, including unused cdiv. The grid header additionally
executes the source num_programs assignment. No ready-register premise or
allocation bound is needed for these memory-free source statements. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

theorem source_count_same (p : Parametric.Parameters) (s : State α) :
    OpSame A M background flatUndef s none (Parametric.countOp p) := by
  unfold Parametric.countOp
  apply op_div A M background flatUndef s none
  · apply op_sub A M background flatUndef s none
    · apply op_add A M background flatUndef s none
      · exact op_constNat A M background flatUndef s none p.size
      · exact op_constNat A M background flatUndef s none p.block
    · exact op_constNat A M background flatUndef s none 1
  · exact op_constNat A M background flatUndef s none p.block

theorem source_before_transport (p : Parametric.Parameters) (s : State α) :
    RunTransport A M background flatUndef (Parametric.before p) s := by
  rw [Parametric.before_exact]
  apply run_cons_transport A M background flatUndef
  · exact assign_alg_data_transport A M background flatUndef .nat [] "pid"
      (.programId 0) s (by decide) (by decide)
      (op_programId A M background flatUndef s none 0)
  · intro u _hu
    apply run_cons_transport A M background flatUndef
    · exact assign_alg_data_transport A M background flatUndef .nat [] "num_tiles0"
        (Parametric.countOp p) u (by decide) (by decide)
        (source_count_same A M background flatUndef p u)
    · intro v _hv
      exact run_nil_transport A M background flatUndef v

theorem source_numCtas_transport (s : State α) :
    StepTransport A M background flatUndef Parametric.numCtas s := by
  exact assign_alg_data_transport A M background flatUndef .nat [] "num_ctas"
    (.numPrograms 0) s (by decide) (by decide)
    (op_numPrograms A M background flatUndef s none 0)

theorem source_header_transport (p : Parametric.Parameters) (s : State α) :
    RunTransport A M background flatUndef (Parametric.header p) s := by
  unfold Parametric.header
  apply run_append_transport A M background flatUndef
  · exact source_before_transport A M background flatUndef p s
  · intro u _hu
    exact run_cons_transport A M background flatUndef Parametric.numCtas [] u
      (source_numCtas_transport A M background flatUndef u)
      (fun v _hv => run_nil_transport A M background flatUndef v)

theorem source_before_flat_run (p : Parametric.Parameters) (s : State α) :
    FP.Structural.run M (flattenComputeStmts A (Parametric.before p))
      (flattenState A background flatUndef s) =
        some (flattenState A background flatUndef (Parametric.beforeState p s)) :=
  source_before_transport A M background flatUndef p s
    (Parametric.beforeState p s) (Parametric.before_eval M p s)

theorem source_header_flat_run (p : Parametric.Parameters) (s : State α) :
    FP.Structural.run M (flattenComputeStmts A (Parametric.header p))
      (flattenState A background flatUndef s) =
        some (flattenState A background flatUndef (Parametric.headerState p s)) :=
  source_header_transport A M background flatUndef p s
    (Parametric.headerState p s) (Parametric.header_eval M p s)

#axiomsClean source_count_same
#axiomsClean source_before_transport
#axiomsClean source_numCtas_transport
#axiomsClean source_header_transport
#axiomsClean source_before_flat_run
#axiomsClean source_header_flat_run

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
