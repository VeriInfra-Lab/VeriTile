import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealPadding
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

inductive ForwardExpected (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ) :
    RegionName → Nat → ℝ → Prop where
  | output (t : Fin s.T) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
      ForwardExpected s f scale bh iv u "o" (valAddress s bh iv s.FBV t.val j.val)
        (Math.output scale (unpaddedColumn s f bh (iv*s.FBV+j.val) u) t.val)
  | residual (t : Fin s.T) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
      ForwardExpected s f scale bh iv u "v" (valAddress s bh iv s.FBV t.val j.val)
        (Math.residual (Math.state (unpaddedColumn s f bh (iv*s.FBV+j.val) u) t.val)
          ((unpaddedColumn s f bh (iv*s.FBV+j.val) u).k t.val) ((unpaddedColumn s f bh (iv*s.FBV+j.val) u).x t.val))
  | terminal (enabled : f.final=Bool.true) (i : Fin s.K) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
      ForwardExpected s f scale bh iv u "ht" (stateAddress s bh iv s.FBV i.val j.val)
        (Math.state (unpaddedColumn s f bh (iv*s.FBV+j.val) u) s.T i)

inductive BackwardExpected (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) :
    RegionName → Nat → ℝ → Prop where
  | query (t : Fin s.T) (i : Fin s.BK) (active : i.val<s.K) :
      BackwardExpected s f scale bh iv a g terminal "dq" (keyAddress s (plane s bh iv) t.val i.val)
        (∑ j : Fin s.BBV,(gradient s scale a g terminal t.val j).q i)
  | key (t : Fin s.T) (i : Fin s.BK) (active : i.val<s.K) :
      BackwardExpected s f scale bh iv a g terminal "dk" (keyAddress s (plane s bh iv) t.val i.val)
        (∑ j : Fin s.BBV,(gradient s scale a g terminal t.val j).k i)
  | value (t : Fin s.T) (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
      BackwardExpected s f scale bh iv a g terminal "dv" (valAddress s bh iv s.BBV t.val j.val)
        (gradient s scale a g terminal t.val j).x
  | vectorBeta (enabled : f.vectorBeta=Bool.true) (t : Fin s.T) (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
      BackwardExpected s f scale bh iv a g terminal "dbeta" (valAddress s (plane s bh iv) iv s.BBV t.val j.val)
        (gradient s scale a g terminal t.val j).beta
  | scalarBeta (enabled : f.vectorBeta=Bool.false) (t : Fin s.T) :
      BackwardExpected s f scale bh iv a g terminal "dbeta" (plane s bh iv*s.T+t.val)
        (∑ j : Fin s.BBV,(gradient s scale a g terminal t.val j).beta)
  | seed (enabled : f.initialGrad=Bool.true) (i : Fin s.BK) (j : Fin s.BBV)
      (key : i.val<s.K) (value : iv*s.BBV+j.val<s.V) :
      BackwardExpected s f scale bh iv a g terminal "dh0" (stateAddress s bh iv s.BBV i.val j.val)
        (Math.carry scale (a j) (fun n => g n j) (terminal j) s.T 0 i)

theorem stored_real (x : ℝ) : (KernelReference.stored algebra x).read .real=x := rfl

theorem forward_complete (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,KernelForward.Facts algebra s f scale bh iv u v ∧
      ∀ r off value,ForwardExpected s f scale bh iv u r off value → (v.mem r off).read .real=value := by
  obtain ⟨v,facts⟩ := KernelForward.complete algebra s f scale bh iv u hv hk hh
  refine ⟨v,facts,?_⟩
  intro r off value expected
  have covered : s.K≤s.BK := Geometry.nextPower_ge s.K
  cases expected with
  | output t j active =>
    rw [facts.outputs t j active,stored_real,forward_output,column_padding s f bh iv u j active,pad_output covered]
  | residual t j active =>
    rw [facts.residuals t j active,stored_real,forward_residual,column_padding s f bh iv u j active]
    unfold Math.residual
    rw [pad_state covered]
    change (unpaddedColumn s f bh (iv*s.FBV+j.val) u).x t.val-
      Math.dot (padVec (Math.state (unpaddedColumn s f bh (iv*s.FBV+j.val) u) t.val))
        (padVec ((unpaddedColumn s f bh (iv*s.FBV+j.val) u).k t.val))=_
    rw [pad_dot covered]
  | terminal enabled i j active =>
    let wide : Fin s.BK := ⟨i.val,lt_of_lt_of_le i.isLt covered⟩
    rw [facts.terminal enabled wide j i.isLt active,stored_real]
    unfold KernelForwardLoop.carry
    rw [forward_state,column_padding s f bh iv u j active,pad_state covered]
    simp [padVec,wide,i.isLt]

theorem backward_complete (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s (KernelBackwardReference.input algebra s f bh iv u) a g terminal)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,KernelBackwardFacts.Facts algebra s f scale bh iv u v ∧
      ∀ r off value,BackwardExpected s f scale bh iv a g terminal r off value → (v.mem r off).read .real=value := by
  obtain ⟨v,facts⟩ := KernelBackwardFacts.complete algebra s f scale bh iv u hv hk hh
  refine ⟨v,facts,?_⟩
  intro r off value expected
  cases expected with
  | query t i active =>
    rw [facts.query t i active,stored_real,query_memory,query_gradient s scale _ a g terminal saved t.val t.isLt]
  | key t i active =>
    rw [facts.key t i active,stored_real,corrected_memory s f scale bh iv u t.val t.isLt i active,
      key_gradient s scale _ a g terminal saved t.val t.isLt]
  | value t j active =>
    rw [facts.value t j active,stored_real]
    unfold KernelBackwardFacts.value KernelBackwardFacts.enhancedAt
    rw [enhanced_time s f scale bh iv u t.val t.isLt,value_gradient s scale _ a g terminal saved]
  | vectorBeta enabled t j active =>
    rw [facts.vectorBeta enabled t j active,stored_real]
    unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
    rw [enhanced_time s f scale bh iv u t.val t.isLt,beta_gradient s scale _ a g terminal saved t.val t.isLt]
  | scalarBeta enabled t =>
    rw [facts.scalarBeta enabled t,stored_real]
    unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
    rw [enhanced_time s f scale bh iv u t.val t.isLt,scalar_beta_gradient s scale _ a g terminal saved t.val t.isLt]
  | seed enabled i j key value =>
    rw [facts.seed enabled i j key value,stored_real]
    unfold KernelReverseMemory.carry
    exact seed_gradient s scale _ a g terminal saved j i

#axiomsClean forward_complete
#axiomsClean backward_complete
#specNonCircular ForwardExpected avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelReference.forward]
#specNonCircular BackwardExpected avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelBackwardReference.reverse]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
