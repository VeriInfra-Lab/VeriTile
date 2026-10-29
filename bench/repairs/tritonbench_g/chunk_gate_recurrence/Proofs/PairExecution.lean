import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem candidate_source_body (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    (candidateSource kind c s seed).surfaceBody = before kind c s seed ++
      [.forRangeDyn "_i" (.constNat 0) (.sub .nat .nil (.constNat c.N) (.constNat 1))
        (.constNat 1) (candidateParts kind c s seed).body] ++ (candidateParts kind c s seed).after := by
  cases kind <;> rfl

private theorem assignment_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (out left right : RegName) (u : State Node) :
    run (engine (algebra isa plan swap)) [KernelOps.assign32 out (KernelOps.update left right)] u =
      run (engine (candidateAlgebra isa plan swap)) [KernelOps.assign32 out (rightUpdate left right)] u := by
  simp only [run,FP.Structural.step,KernelOps.assign32,evalExpr,evalComputeOp]
  exact congrArg (fun (e : Option (Values Node .real [64,16])) =>
    (e.bind (fun values => some (u.setReg out .real [64,16] values))).bind some)
    (ProgramPair.eval_update isa plan swap left right u)

theorem after_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    run (engine (algebra isa plan swap)) (after kind c s seed) u =
      run (engine (candidateAlgebra isa plan swap)) (candidateParts kind c s seed).after u := by
  cases kind with
  | forward =>
    change run _ [] u = run _ [] u
    simp only [run]
  | backward =>
    have hp := funext (tail_before_run isa plan swap c s seed)
    have ht := funext (tail_after_run isa plan swap c s seed)
    have ha := funext (assignment_run isa plan swap "Dacc" "Dacc" "DS_i")
    simp only [tail_original,tail_candidate,run_append,hp,ht,ha]

theorem complete_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    FP.Structural.exec (engine (algebra isa plan swap)) (source kind c s seed) u =
      FP.Structural.exec (engine (candidateAlgebra isa plan swap)) (candidateSource kind c s seed) u := by
  have hp := funext (before_run isa plan swap kind c s seed)
  have ht := funext (after_run isa plan swap kind c s seed)
  have hl : run (engine (algebra isa plan swap)) [loop kind c s seed] =
      run (engine (candidateAlgebra isa plan swap))
        [.forRangeDyn "_i" (.constNat 0) (.sub .nat .nil (.constNat c.N) (.constNat 1))
          (.constNat 1) (candidateParts kind c s seed).body] := by
    funext v
    simpa only [run,KernelProgram.loop,FP.Structural.step,FP.Structural.evalOp,
      Option.bind_some,bind] using
      congrArg (fun e : Option (State Node) => e.bind some)
        (range_run isa plan swap kind c s seed (c.N-1) 1 0 v)
  change run _ (source kind c s seed).surfaceBody u = run _ (candidateSource kind c s seed).surfaceBody u
  simp only [source_body,candidate_source_body,run_append,hp,ht,hl]

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
