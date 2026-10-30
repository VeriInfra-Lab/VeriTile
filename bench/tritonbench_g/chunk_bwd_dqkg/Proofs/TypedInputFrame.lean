import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedTile
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def SameInputs (s t : State α) : Prop :=
  ∀ r,r ∈ (["q","k","v","h","g","do","dh"] : List RegionName) → s.mem r=t.mem r

def inputState (s : State α) : State α :=
  {s with mem:=fun r=>if r ∈ (["q","k","v","h","g","do","dh"] : List RegionName) then s.mem r else fun _=>.mk .real default
          regs:=fun _ _ _=>none
          undef:=fun _ _ _=>defaultValue _}

theorem values_inputState (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (n : Nat) :
    accumulated M dot c (inputState s) n=accumulated M dot c s n := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    simp only [accumulated,ih]
    simp [updateCarries,valTile,histTile,inputState]
    exact ⟨rfl,rfl,rfl,rfl⟩

theorem answer_inputState (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : Carries α c) :
    answerQ M dot c (inputState s) a=answerQ M dot c s a ∧
    answerK M dot c (inputState s) a=answerK M dot c s a ∧
    answerG M dot c (inputState s) a=answerG M dot c s a := by
  simp [answerQ,answerK,answerG,gateContribution,lastCorrection,gatedQ,gatedK,scores,
    keyTile,gate,gateLast,inputState]
  exact ⟨rfl,rfl,rfl⟩

theorem inputState_prepare_eq (c : Config) (s t : State α) (a : CTA c) (h : SameInputs s t) :
    inputState (prepare c s a)=inputState (prepare c t a) := by
  have hm : (inputState (prepare c s a)).mem=(inputState (prepare c t a)).mem := by
    funext r off
    simp only [inputState,prepare]
    split_ifs with hr
    · exact congrFun (h r hr) off
    · rfl
  cases s;cases t
  simp only [inputState,prepare] at hm ⊢
  congr 1

theorem results_inputs (M : Algebra α) (dot : Dot α) (c : Config) (s t : State α)
    (a : CTA c) (h : SameInputs s t) :
    resultQ M dot c s a=resultQ M dot c t a ∧
    resultK M dot c s a=resultK M dot c t a ∧
    resultG M dot c s a=resultG M dot c t a := by
  have hs := values_inputState M dot c (prepare c s a) (valueBlocks c)
  have ht := values_inputState M dot c (prepare c t a) (valueBlocks c)
  have he := inputState_prepare_eq c s t a h
  have ha : accumulated M dot c (prepare c s a) (valueBlocks c)=accumulated M dot c (prepare c t a) (valueBlocks c) :=
    hs.symm.trans ((congrArg (fun u=>accumulated M dot c u (valueBlocks c)) he).trans ht)
  have hq := (answer_inputState M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c))).1
  have hk := (answer_inputState M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c))).2.1
  have hg := (answer_inputState M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c))).2.2
  rw [he,ha] at hq hk hg
  rw [(answer_inputState M dot c (prepare c t a) _).1] at hq
  rw [(answer_inputState M dot c (prepare c t a) _).2.1] at hk
  rw [(answer_inputState M dot c (prepare c t a) _).2.2] at hg
  exact ⟨by unfold resultQ;rw [ha,←hq],by unfold resultK;rw [ha,←hk],by unfold resultG;rw [ha,←hg]⟩

theorem tile_inputs (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c) :
    SameInputs (finalState M dot c (prepare c s a)) s := by
  intro region hr
  funext off
  exact tile_frame M dot c s a region off (input_unwritten c a region off hr)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
