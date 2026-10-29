import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Host

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host
noncomputable section
theorem Completion.caller_frame {α β σ : Type} {ops : Source.Arithmetic α β} {runtime : Runtime σ}
    {live : σ → Nat → Prop} {args : Arguments} {initial final : World α σ} {output : Tensor}
    (e : Completion ops runtime live args initial final output)
    (point : Address) (old : live initial.native point.1) : final.memory point=initial.memory point := by
  have fresh := e.fresh point.1 old
  have outside : ∀ row<args.problem.M,∀ column<args.problem.N,
      output.address row column≠point := by
    intro row _ column _ he
    exact fresh (congrArg Prod.fst he)
  cases e.branch with
  | empty empty same => rw [same]
  | zero nonempty zeroK native executed same =>
    rw [same]
    change zeroMemory ops.inputZero args.problem ⟨args.A,args.B,output⟩ initial.memory point=_
    unfold zeroMemory
    rw [if_neg]
    rintro ⟨row,hr,column,hc,he⟩
    exact outside row hr column hc he
  | launch nonempty positiveK trials selected executed =>
    have memory := (sequence_memory ops runtime args output (trials+1) _ final executed).1
    rw [memory,Source.replay_positive ops args.problem _ e.layout trials initial.memory]
    exact Source.source_frame ops args.problem _ e.layout initial.memory point outside

#axiomsClean Completion.caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host
