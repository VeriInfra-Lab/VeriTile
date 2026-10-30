import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeLaunch
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostSetup
noncomputable section
inductive Call where
  | validateRanks | validateShapes | validateDevices
  | resolveQuery | resolveKey | resolveValue
  | resolveQueryScale | packQueryScale | resolveKeyScale | packKeyScale
  | validateHeads | allocateOutput
  | validateQueryScales | validateKeyScales | compileAndLaunch
  deriving DecidableEq, Repr

def beforeEmpty : List Call :=
  [.validateRanks,.validateShapes,.validateDevices,.resolveQuery,.resolveKey,.resolveValue,
   .resolveQueryScale,.packQueryScale,.resolveKeyScale,.packKeyScale,.validateHeads,.allocateOutput]

def nonemptyChecks : List Call := [.validateQueryScales,.validateKeyScales]

def calls (ok : Call → Bool) : List Call → Except Call Unit
  | [] => .ok ()
  | c::cs => if ok c then calls ok cs else .error c

theorem first_failure (ok : Call → Bool) (xs : List Call) (failed : Call)
    (h : calls ok xs=.error failed) :
    ∃ before after, xs=before++failed::after ∧ (∀ c∈before,ok c=Bool.true) ∧ ok failed=Bool.false := by
  induction xs with
  | nil => simp [calls] at h
  | cons c cs ih =>
    cases hc : ok c with
    | «false» =>
      simp [calls,hc] at h
      subst failed
      exact ⟨[],cs,rfl,by simp,hc⟩
    | «true» =>
      simp [calls,hc] at h
      obtain ⟨before,after,he,hb,hf⟩ := ih h
      refine ⟨c::before,after,by simp [he],?_,hf⟩
      intro a ha
      rcases List.mem_cons.mp ha with rfl | ha
      · exact hc
      · exact hb a ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostSetup
