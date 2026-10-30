import bench.tritonbench_g.rotary_emb.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Reuse the complete established source result; bind all four output
channels and retain the exact exterior cell frame through the shared IO. -/
theorem invocation_result {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat)
    (values : Inputs A shape) (s : State A)
    (bound : (kernelIO rest ops shape pids).InputBinding values s) :
    ∃ t,(invocation rest ops shape pids).run s=some t ∧
      (kernelIO rest ops shape pids).OutputBinding values (rotation ops shape values) t ∧
      (kernelIO rest ops shape pids).Frame values s t := by
  obtain ⟨t,executed,_,outputs,frame⟩ :=
    Typed.successful rest ops shape valid {s with pids := pids}
  have readback (key odd : Bool) (j : Lane shape) (live : payloadActive shape pids key j) :
      t.mem (if key then "K" else "Q") (payloadOffset shape pids key odd j)=
        Cell.mk .real (pairFormula ops values key odd j) := by
    have result := outputs key odd j live
    rw [value_pairFormula rest ops shape pids values s bound key odd j live,offset_eq] at result
    exact result
  refine ⟨t,?_,?_,?_⟩
  · simpa only [Invocation.run,invocation_initial,invocation] using executed
  · intro o j live
    fin_cases o
    all_goals dsimp [kernelIO,memory,rotation,outputChannels] at live ⊢
    · exact congrArg (mapCell id) (readback Bool.false Bool.false j live)
    · exact congrArg (mapCell id) (readback Bool.false Bool.true j live)
    · exact congrArg (mapCell id) (readback Bool.true Bool.false j live)
    · exact congrArg (mapCell id) (readback Bool.true Bool.true j live)
  · intro r offset outside
    exact frame r offset (outside_typed rest ops shape pids values s r offset outside)

theorem nativeRotation_eq (p : Native.Instructions) (shape : Shape) :
    rotation (Native.ops p) shape=nativeRotation p shape := by
  funext values o j
  fin_cases o <;> rfl

#axiomsClean invocation_result
#axiomsClean nativeRotation_eq
end
end VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
