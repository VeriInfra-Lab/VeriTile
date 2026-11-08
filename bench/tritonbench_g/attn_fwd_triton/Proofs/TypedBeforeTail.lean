import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedAddress
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedPreservation
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem middle_assembled (p : Config) (BD : Nat) :
    middleCode p BD=initializationCode p BD ++ setupCode p BD Bool.false ++
      loopCode p BD Bool.false ++ setupCode p BD Bool.true ++ loopCode p BD Bool.true := by
  cases hd : decide (p.stage=2)
  all_goals simp only [middleCode,prefixLength,coordinateCode,pointerCode,initializationCode,setupCode,setup,preloop,loopCode,finish,body,parts,workName,Source.intRange?,Option.map_some,Option.getD_some,beq_self_eq_true,
    lowerOp,lowerName,upperName,code,source,TypedSource.repaired,originalKernel,ComputeKernel.surfaceBody,hd,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ControlView.compute,
    List.length_cons,List.length_nil,List.cons_append,List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero]
  all_goals rfl

theorem middle_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (s : State α)
    (hs : PointerReady BD s) :
    ∃ t, run (engine A) (middleCode p BD) s=some t ∧ Ready BD t := by
  obtain ⟨s3,h3,hr3⟩ := initialization_success A p BD s hs
  obtain ⟨s4,h4,hr4,hl4⟩ := setup_success A p BD Bool.false s3 hr3
  obtain ⟨s5,h5,hr5⟩ := loop_success A p BD support Bool.false s4 hr4 hl4
  obtain ⟨s6,h6,hr6,hl6⟩ := setup_success A p BD Bool.true s5 hr5
  obtain ⟨s7,h7,hr7⟩ := loop_success A p BD support Bool.true s6 hr6 hl6
  refine ⟨s7,?_,hr7⟩
  rw [middle_assembled]
  simp only [run_append,h3,h4,h5,h6,h7,Option.bind_some]

theorem prefix_assembled (p : Config) (BD : Nat) :
    (code p BD).take (prefixLength p BD)=(code p BD).take 22 ++ middleCode p BD := by
  have h : 22≤prefixLength p BD := by
    cases hd : decide (p.stage=2)
    all_goals simp only [prefixLength,code,source,TypedSource.repaired,originalKernel,
      hd,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
      List.length_cons,List.length_nil]
    all_goals decide
  simpa only [middleCode,Nat.add_sub_of_le h] using
    (List.take_add (l:=code p BD) (i:=22) (j:=prefixLength p BD-22))

theorem before_tail {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (r : Bool) (s : State α) :
    ∃ t, run (engine A) ((code p BD).take (prefixLength p BD)) s=some t ∧
      Ready BD t ∧ AddressReady p BD (s.pids 0) (s.pids 1) t ∧ t.mem=s.mem := by
  obtain ⟨u,hu,hr,ha,hm⟩ := header A p BD s
  obtain ⟨t,ht,hready⟩ := middle_success A p BD support u hr
  have hk := pure_keep (engine A) ["n_ctx","offs_m","O_block_ptr"] _ (middle_pure p BD) u t ht
  refine ⟨t,?_,hready,?_,hk.1.trans hm⟩
  · rw [prefix_assembled]
    simp only [run_append,hu,ht,Option.bind_some]
  · rcases ha with ⟨hn,ho,hp⟩
    exact ⟨(hk.2 _ _ _ (by simp)).trans hn,
      (hk.2 _ _ _ (by simp)).trans ho,(hk.2 _ _ _ (by simp)).trans hp⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
