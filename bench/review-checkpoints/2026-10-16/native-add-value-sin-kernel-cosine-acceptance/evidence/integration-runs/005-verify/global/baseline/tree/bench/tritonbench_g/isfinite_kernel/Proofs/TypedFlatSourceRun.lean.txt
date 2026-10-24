import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSourceBody
import bench.tritonbench_g.isfinite_kernel.Proofs.BitCorrect

/-! Complete original source transport, including header, both branches and
the actual grid loop. The local eight-statement proof discharges each body
obligation; source execution is never a transport hypothesis. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

def sourceProgramBody (p : Parametric.Parameters) (wide : Bool)
    (loopTile : Op .nat [] := sourceLoopTile) : List ComputeStmt :=
  Parametric.before p ++
    [.ifThenElse (.constBool p.oneTile) (sourceBody p wide (.ref .nat [] "pid"))
      [Parametric.numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
        (.constNat 1) (sourceBody p wide loopTile)]]

theorem source32_body_exact (p : Parametric.Parameters) :
    (Parametric.original32 p).surfaceBody = sourceProgramBody p Bool.false := by
  rw [Parametric.source32_split, source_one32_exact, source_loop32_exact]
  rfl
theorem source64_body_exact (p : Parametric.Parameters) :
    (Parametric.original64 p).surfaceBody = sourceProgramBody p Bool.true := by
  rw [Parametric.source64_split, source_one64_exact, source_loop64_exact]
  rfl

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

theorem source_program_transport (hd : A.Disjoint) (p : Parametric.Parameters)
    (wide : Bool) (loopTile : Op .nat [])
    (hloop : ∀ state : State α, OpSame A M background flatUndef state none loopTile)
    (s : State α) (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output) :
    RunTransport A M background flatUndef (sourceProgramBody p wide loopTile) s := by
  unfold sourceProgramBody
  refine run_append_transport A M background flatUndef _ _ s
    (source_before_transport A M background flatUndef p s) ?_
  intro prefixState hp
  rw [Parametric.before_eval] at hp
  have he := Option.some.inj hp
  subst prefixState
  have hprefix : UndefCompatible A flatUndef (Parametric.beforeState p s) .real := hu
  refine run_cons_transport A M background flatUndef _ _ _ ?_
    (fun t _ => run_nil_transport A M background flatUndef t)
  apply const_if_transport A M background flatUndef
  · exact source_body_transport A M background flatUndef hd p wide _ _
      (op_ref_data A M background flatUndef _ none .nat [] "pid" (by decide) (by decide))
      hprefix hinput houtput hin hout
  · refine run_cons_transport A M background flatUndef _ _ _
      (source_numCtas_transport A M background flatUndef _) ?_
    intro header hh
    have hheader := assign_compatible A M flatUndef .nat [] "num_ctas" (.alg (.numPrograms 0))
      (Parametric.beforeState p s) header hh hprefix
    refine run_cons_transport A M background flatUndef _ _ header ?_
      (fun t _ => run_nil_transport A M background flatUndef t)
    have hbody (j : Nat) (state : State α) (hs : UndefCompatible A flatUndef state .real) :
        RunTransport A M background flatUndef (sourceBody p wide loopTile)
          (state.setReg "j" .nat [] (fun _ => j)) :=
      source_body_transport A M background flatUndef hd p wide loopTile _
        (hloop _) hs hinput houtput hin hout
    have hpreserve (j : Nat) (state final : State α)
        (hs : UndefCompatible A flatUndef state .real)
        (hr : FP.Structural.run M (sourceBody p wide loopTile)
          (state.setReg "j" .nat [] (fun _ => j)) = some final) :
        UndefCompatible A flatUndef final .real := by
      unfold UndefCompatible at *
      rw [source_body_undef M p wide loopTile _ final hr]
      exact hs
    simpa only [Nat.zero_add] using
      const_range_transport A M background flatUndef "j" (sourceBody p wide loopTile)
        (fun state => UndefCompatible A flatUndef state .real)
        hbody hpreserve p.iterations 0 header hheader

theorem exec_of_transport (kernel : ComputeKernel) (s t : State α)
    (h : RunTransport A M background flatUndef kernel.surfaceBody s)
    (ht : FP.Structural.exec M kernel s = some t) :
    FP.Structural.exec M (flattenComputeKernel A kernel) (flattenState A background flatUndef s) =
      some (flattenState A background flatUndef t) := by
  cases kernel with
  | mk inputs outputs body =>
      exact h t ht

theorem original32_flat_exec (hd : A.Disjoint) (p : Parametric.Parameters)
    (s t : State α) (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (ht : FP.Structural.exec M (Parametric.original32 p) s = some t) :
    FP.Structural.exec M (flattenComputeKernel A (Parametric.original32 p))
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t) := by
  apply exec_of_transport A M background flatUndef _ s t ?_ ht
  rw [source32_body_exact]
  exact source_program_transport A M background flatUndef hd p Bool.false sourceLoopTile
    (source_loop_tile_same A M background flatUndef) s hu hinput houtput hin hout

theorem original64_flat_exec (hd : A.Disjoint) (p : Parametric.Parameters)
    (s t : State α) (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (ht : FP.Structural.exec M (Parametric.original64 p) s = some t) :
    FP.Structural.exec M (flattenComputeKernel A (Parametric.original64 p))
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t) := by
  apply exec_of_transport A M background flatUndef _ s t ?_ ht
  rw [source64_body_exact]
  exact source_program_transport A M background flatUndef hd p Bool.true sourceLoopTile
    (source_loop_tile_same A M background flatUndef) s hu hinput houtput hin hout

#axiomsClean source32_body_exact
#axiomsClean source64_body_exact
#axiomsClean source_program_transport
#axiomsClean exec_of_transport
#axiomsClean original32_flat_exec
#axiomsClean original64_flat_exec
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
