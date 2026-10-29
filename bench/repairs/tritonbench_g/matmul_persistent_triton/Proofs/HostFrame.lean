import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Host
noncomputable section
theorem Completion.caller_frame {α β σ : Type} {ops : Source.Arithmetic α β} {runtime : Runtime σ}
    {live : σ → Nat → Prop} {args : Arguments} {initial final : World α σ} {output : Tensor}
    (e : Completion ops runtime live args initial final output)
    (point : Address) (old : live initial.native point.1) : final.memory point=initial.memory point := by
  have fresh := e.fresh point.1 old
  have outside : ∀ row<(args.problem e.sms).M,∀ column<(args.problem e.sms).N,
      output.address row column≠point := by
    intro row _ column _ he
    exact fresh (congrArg Prod.fst he)
  cases e.branch with
  | empty empty same => rw [same]
  | zero nonempty zeroK native executed same =>
    rw [same]
    change zeroMemory ops.inputZero (args.problem e.sms) ⟨args.A,args.B,output⟩ initial.memory point=_
    unfold zeroMemory
    rw [if_neg]
    rintro ⟨row,hr,column,hc,he⟩
    exact outside row hr column hc he
  | launch nonempty positiveK native executed same =>
    rw [same]
    exact Source.source_frame ops (args.problem e.sms) _ e.layout initial.memory point outside

#axiomsClean Completion.caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Host
