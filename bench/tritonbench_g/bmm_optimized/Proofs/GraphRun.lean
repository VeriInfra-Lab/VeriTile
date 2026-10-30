import bench.tritonbench_g.bmm_optimized.Proofs.GraphExecution

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- The actual source body's common range executes identically at every reached
state. The live graph carry is retained through all iterations. -/
theorem range_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (idx : RegName) (start stop stride : Nat) (s : State (Node α)) :
    Execution.range (scalar p) (base p rest) idx start stop stride (Source.body cfg) s=
      FP.Structural.range (engine p rest) idx start stop stride (Source.body cfg) s := by
  by_cases hz : stride=0
  · rw [Execution.range,FP.Structural.range,if_pos hz,if_pos hz]
  · have aux : ∀ fuel start (s : State (Node α)),stop-start=fuel →
        Execution.range (scalar p) (base p rest) idx start stop stride (Source.body cfg) s=
          FP.Structural.range (engine p rest) idx start stop stride (Source.body cfg) s := by
      intro fuel
      induction fuel using Nat.strong_induction_on with
      | h fuel ih=>
        intro start s hf
        rw [Execution.range,FP.Structural.range,if_neg hz,if_neg hz]
        by_cases active : start<stop
        · rw [if_pos active,if_pos active,body_equal]
          cases run : FP.Structural.run (engine p rest) (Source.body cfg)
              (s.setReg idx .nat [] (fun _=>start)) with
          | none=>rfl
          | some next=>exact ih (stop-(start+stride)) (by omega) (start+stride) next rfl
        · rw [if_neg active,if_neg active]
    exact aux _ start s rfl

theorem loop_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (s : State (Node α)) :
    Execution.run (scalar p) (base p rest)
        [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1) (Source.body cfg)] s=
      FP.Structural.run (engine p rest)
        [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1) (Source.body cfg)] s := by
  simp only [Execution.run,FP.Structural.run,Execution.step,FP.Structural.step]
  change ((evalOp (engine p rest) none (.constNat 0) s).bind (fun a=>
      (evalOp (engine p rest) none (.ref .nat [] "num_iters") s).bind (fun b=>
        (evalOp (engine p rest) none (.constNat 1) s).bind (fun c=>
          Execution.range (scalar p) (base p rest) "_i" (a PUnit.unit) (b PUnit.unit)
            (c PUnit.unit) (Source.body cfg) s)))).bind some = _
  simp only [range_equal]
  rfl

/-- Exact original source execution, with no replacement AST or numerical law. -/
theorem exec_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (s : State (Node α)) :
    Execution.exec (scalar p) (base p rest) cfg s=
      FP.Structural.exec (engine p rest) (Source.kernel cfg) s := by
  change Execution.run (scalar p) (base p rest) (Source.kernel cfg).surfaceBody s=
    FP.Structural.run (engine p rest) (Source.kernel cfg).surfaceBody s
  rw [Source.source_decomposition,Execution.run_append,Execution.run_append,
    FP.Structural.run_append,FP.Structural.run_append,prelude_equal]
  simp only [loop_equal,epilogue_equal]

#axiomsClean range_equal
#axiomsClean loop_equal
#axiomsClean exec_equal
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
