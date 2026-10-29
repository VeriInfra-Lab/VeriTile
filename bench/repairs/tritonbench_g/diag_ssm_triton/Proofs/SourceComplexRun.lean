import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceLoopSimulation
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexStepRelation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem forward_run_related (p : ForwardLayout) (length : Nat) (u : BlockState) :
    ∃ va vb,
      exec (Kernels.complexForwardTyped p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType.toAlgKernel u = some va ∧
      exec (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel u = some vb ∧
      ForwardRelated p length va vb := by
  let a := (forwardPrelude p length u).val
  let b := (unsignedPrelude p u).val
  have hi : ForwardRelated p length a b := prelude_related p length u
  obtain ⟨va, vb, ha, hb, hr⟩ := VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal.paired_loop "t" length (forwardLoop p length)
    (ComplexForwardSource.body p.x p.y p.batch p.dim p.B)
    (ForwardRelated p length) (fun i a b _ h => forward_step_related p length i a b h)
    0 (Nat.zero_le length) a b hi
  refine ⟨va, vb, ?_, ?_, hr⟩
  · change stepStmts (forwardBody p length) u = some va
    rw [forward_decomposition, ProgramRewrite.append, (forwardPrelude p length u).property]
    simp only [Option.bind_some, stepStmts, stepForRangeIntAux.forRangeInt_unfold,
      evalOp, hi.lengthReg, Tile.scalar, Option.bind_some]
    simpa only [a, forwardPrelude_length, Tile.scalar, Option.bind_some, Int.natCast_zero] using
      congrArg (fun o : Option BlockState => match o with | some v => some v | none => none) ha
  · unfold exec
    rw [ComplexForwardSource.source_body]
    simp only [ProgramRewrite.append,
      (unsignedPrelude p u).property, Option.bind_some, stepStmts, stepForLoopAux.forLoop_unfold]
    exact congrArg (fun o : Option BlockState => match o with | some v => some v | none => none) hb

/-- The actual copied source produces the same memory as the independent
mathematical loop; temporary integer registers need not coincide. -/
theorem forward_memory (p : ForwardLayout) (length : Nat) (u old : BlockState)
    (ho : exec (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel u = some old) :
    ∃ actual, exec (Kernels.complexForwardTyped p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType.toAlgKernel u = some actual ∧
      actual.mem = old.mem := by
  obtain ⟨a, b, ha, hb, h⟩ := forward_run_related p length u
  have he : b = old := Option.some.inj (hb.symm.trans ho)
  subst old
  exact ⟨a, ha, h.memory⟩

#axiomsClean forward_run_related
#axiomsClean forward_memory
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
