import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardCompose
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPost
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem backward_run_memory (p : BackwardLayout) (length : Nat) (u : BlockState) :
    ∃ va vb,
      exec (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel u = some va ∧
      exec (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel u = some vb ∧
      va.mem = vb.mem := by
  let a := (forwardPrelude p length u).val
  let b := (unsignedPrelude p u).val
  have hi : BackwardRelated p length a b := prelude_related p length u
  obtain ⟨va, vb, ha, hb, hr⟩ := SourceReal.paired_loop "i" length (forwardLoop p length)
    (historicalLoop p length) (BackwardRelated p length)
    (fun i a b ht h => backward_step_related p length i a b ht h)
    0 (Nat.zero_le length) a b hi
  simp only [a, Int.natCast_zero] at ha
  simp only [b, historicalLoop] at hb
  obtain ⟨aa, bb, hpa, hpb, hm⟩ := post_memory p length va vb hr
  refine ⟨aa, bb, ?_, ?_, hm⟩
  · change stepStmts (forwardBody p length) u = some aa
    rw [forward_decomposition]
    simp only [List.append_assoc, ProgramRewrite.append, (forwardPrelude p length u).property,
      Option.bind_some, stepStmts, stepForRangeIntAux.forRangeInt_unfold,
      evalOp, forwardPrelude_length, Tile.scalar, Option.bind_some]
    rw [ha]
    exact hpa
  · unfold exec
    rw [ComplexBackwardSource.source_body]
    simp only [List.append_assoc, ProgramRewrite.append,
      (unsignedPrelude p u).property, Option.bind_some, stepStmts, stepForLoopAux.forLoop_unfold]
    rw [hb]
    exact hpb

theorem backward_memory (p : BackwardLayout) (length : Nat) (u old : BlockState)
    (ho : exec (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel u = some old) :
    ∃ actual, exec (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel u = some actual ∧
      actual.mem = old.mem := by
  obtain ⟨a, b, ha, hb, h⟩ := backward_run_memory p length u
  have he : b = old := Option.some.inj (hb.symm.trans ho)
  subst old
  exact ⟨a, ha, h⟩

#axiomsClean backward_run_memory
#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
