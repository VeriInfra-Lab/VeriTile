import bench.tritonbench_g.fused_recurrent_delta.Proofs.InputAgreement
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOInput
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_data (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : ForwardInputs α s) (u : State α)
    (hin : (forwardIO M s f scale bh iv).InputBinding vals u) :
    InputAgreement.Forward (KernelReference.forwardInput M s f bh iv u) (forwardData M s f iv vals) := by
  constructor
  · intro t ht
    funext i
    by_cases active : i.val<s.K
    · have read := hin (1 : Fin 6) (⟨t,ht⟩,i,()) active
      change (u.mem "k" (keyAddress s bh t i.val)).read .real=vals (1 : Fin 6) (⟨t,ht⟩,i,()) at read
      simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active,read]
    · simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active]
  · intro t ht
    funext i
    by_cases active : i.val<s.K
    · have read := hin (0 : Fin 6) (⟨t,ht⟩,i,()) active
      change (u.mem "q" (keyAddress s bh t i.val)).read .real=vals (0 : Fin 6) (⟨t,ht⟩,i,()) at read
      simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active,read]
    · simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active]
  · intro t ht
    funext j
    by_cases active : iv*s.FBV+j.val<s.V
    · have read := hin (2 : Fin 6) (⟨t,ht⟩,j,()) active
      change (u.mem "v" (valAddress s bh iv s.FBV t j.val)).read .real=vals (2 : Fin 6) (⟨t,ht⟩,j,()) at read
      simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active,read]
    · simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,active]
  · intro t ht
    funext j
    cases flag : f.vectorBeta
    · have read := hin (4 : Fin 6) (⟨t,ht⟩,()) flag
      change (u.mem "beta" (betaAddress s bh t)).read .real=vals (4 : Fin 6) (⟨t,ht⟩,()) at read
      simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,flag,read]
    · by_cases active : iv*s.FBV+j.val<s.V
      · have read := hin (3 : Fin 6) (⟨t,ht⟩,j,()) ⟨flag,active⟩
        change (u.mem "beta" (valAddress s bh iv s.FBV t j.val)).read .real=vals (3 : Fin 6) (⟨t,ht⟩,j,()) at read
        simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,flag,active,read]
      · simp [KernelReference.forwardInput,KernelReference.read,forwardData,ht,flag,active]
  · funext ji
    by_cases enabled : f.initial=Bool.true
    · by_cases hk : ji.2.1.val<s.K
      · by_cases hv : iv*s.FBV+ji.1.val<s.V
        · have read := hin (5 : Fin 6) (ji.2.1,ji.1,()) ⟨enabled,hk,hv⟩
          change (u.mem "h0" (stateAddress s bh iv s.FBV ji.2.1.val ji.1.val)).read .real=
            vals (5 : Fin 6) (ji.2.1,ji.1,()) at read
          simp [KernelReference.forwardInput,KernelReference.read,forwardData,enabled,hk,hv,read]
        · simp [KernelReference.forwardInput,KernelReference.read,forwardData,enabled,hk,hv]
      · simp [KernelReference.forwardInput,KernelReference.read,forwardData,enabled,hk]
    · simp [KernelReference.forwardInput,KernelReference.read,forwardData,enabled]

theorem backward_data (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : BackwardInputs α s) (u : State α)
    (hin : (backwardIO M s f scale bh iv).InputBinding vals u) :
    InputAgreement.Backward (KernelBackwardReference.input M s f bh iv u) (backwardData M s f iv vals) := by
  constructor
  · intro t ht
    funext i
    by_cases active : i.val<s.K
    · have read := hin (1 : Fin 8) (⟨t,ht⟩,i,()) active
      change (u.mem "k" (keyAddress s bh t i.val)).read .real=vals (1 : Fin 8) (⟨t,ht⟩,i,()) at read
      simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active,read]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active]
  · intro t ht
    funext i
    by_cases active : i.val<s.K
    · have read := hin (0 : Fin 8) (⟨t,ht⟩,i,()) active
      change (u.mem "q" (keyAddress s bh t i.val)).read .real=vals (0 : Fin 8) (⟨t,ht⟩,i,()) at read
      simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active,read]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active]
  · intro t ht
    funext j
    by_cases active : iv*s.BBV+j.val<s.V
    · have read := hin (2 : Fin 8) (⟨t,ht⟩,j,()) active
      change (u.mem "v" (valAddress s bh iv s.BBV t j.val)).read .real=vals (2 : Fin 8) (⟨t,ht⟩,j,()) at read
      simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active,read]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active]
  · intro t ht
    funext j
    cases flag : f.vectorBeta
    · have read := hin (4 : Fin 8) (⟨t,ht⟩,()) flag
      change (u.mem "beta" (betaAddress s bh t)).read .real=vals (4 : Fin 8) (⟨t,ht⟩,()) at read
      simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,flag,read]
    · by_cases active : iv*s.BBV+j.val<s.V
      · have read := hin (3 : Fin 8) (⟨t,ht⟩,j,()) ⟨flag,active⟩
        change (u.mem "beta" (valAddress s bh iv s.BBV t j.val)).read .real=vals (3 : Fin 8) (⟨t,ht⟩,j,()) at read
        simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,flag,active,read]
      · simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,flag,active]
  · intro t ht
    funext j
    by_cases active : iv*s.BBV+j.val<s.V
    · have read := hin (5 : Fin 8) (⟨t,ht⟩,j,()) active
      change (u.mem "do" (valAddress s bh iv s.BBV t j.val)).read .real=vals (5 : Fin 8) (⟨t,ht⟩,j,()) at read
      simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active,read]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,ht,active]
  · funext ji
    by_cases enabled : f.initial=Bool.true
    · by_cases hk : ji.1.val<s.K
      · by_cases hv : iv*s.BBV+ji.2.1.val<s.V
        · have read := hin (6 : Fin 8) ji ⟨enabled,hk,hv⟩
          change (u.mem "h0" (stateAddress s bh iv s.BBV ji.1.val ji.2.1.val)).read .real=
            vals (6 : Fin 8) ji at read
          simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk,hv,read]
        · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk,hv]
      · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled]
  · funext ji
    by_cases enabled : f.terminalGrad=Bool.true
    · by_cases hk : ji.1.val<s.K
      · by_cases hv : iv*s.BBV+ji.2.1.val<s.V
        · have read := hin (7 : Fin 8) ji ⟨enabled,hk,hv⟩
          change (u.mem "dht" (stateAddress s bh iv s.BBV ji.1.val ji.2.1.val)).read .real=
            vals (7 : Fin 8) ji at read
          simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk,hv,read]
        · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk,hv]
      · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled,hk]
    · simp [KernelBackwardReference.input,KernelReference.read,backwardData,enabled]

#axiomsClean forward_data
#axiomsClean backward_data
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOInput
