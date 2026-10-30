import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOValues
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.RealPublic

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def hostCounts (c : Host.Call) (x : Host.Input A) : Fin c.batch → Nat := fun b => x.counts b.val

/-- Every old finite-batch input supplies its own finite covering extent.
There is no `counts ≤ maximum` premise, including for decoded native words. -/
theorem hostCounts_covered (c : Host.Call) (x : Host.Input A) (b : Fin c.batch) :
    x.counts b.val ≤ capacity (hostCounts c x) := count_le_capacity (hostCounts c x) b

def hostInputs (c : Host.Call) (x : Host.Input A) : Inputs A c (hostCounts c x) :=
  fun port => match port with
  | ⟨0,_⟩ => fun j => x.counts j.1.val
  | ⟨1,_⟩ => fun j => x.starts j.1.val
  | ⟨2,_⟩ => fun j => Int.ofNat (x.pages (j.1.val*c.maximum+(c.maximum-x.counts j.1.val)+j.2.1.val))
  | ⟨3,_⟩ => fun j => x.logics (j.2.1.val*c.tokens+(x.starts j.1.val+j.2.2.1.val))
  | ⟨4,_⟩ => fun j => x.values (j.2.1.val*c.dimensions+j.2.2.2.1.val+
      x.pages (j.1.val*c.maximum+(c.maximum-x.counts j.1.val)+j.2.2.1.val)*(c.heads*c.dimensions))
  | ⟨k+5,hp⟩ => False.elim (by simp [inputs] at hp)

theorem hostInputs_match (c : Host.Call) (x : Host.Input A) : LengthsMatch (hostInputs c x) := by
  intro b _
  rfl

def mapInput (encode : A → B) (x : Host.Input A) : Host.Input B :=
  ⟨fun a => encode (x.logics a),fun a => encode (x.values a),x.starts,x.counts,x.pages⟩

theorem hostInputs_binding [Inhabited B] (M : Host.Job → Algebra B) (encode : A → B)
    (c : Host.Call) (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (work : List Host.Job) (x : Host.Input A) (out : Nat → B) (j : Host.Job) :
    (sourceIO M encode c observe (hostCounts c x) work).InputBinding (hostInputs c x)
      (Host.launch c (mapInput encode x) out j) := by
  intro port i _
  fin_cases port <;>
    simp [sourceIO,memory,inputs,hostInputs,read,Host.launch,mapInput,mapValue,
      start,page,hostCounts]
  all_goals try simp only [←neg_mul,←Int.natCast_mul,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]
  all_goals rfl

theorem host_logit [Inhabited A] (c : Host.Call) (x : Host.Input A)
    (b : Fin c.batch) (h : Fin c.heads) (k : Nat) (hk : k < x.counts b.val) :
    logitAt (hostInputs c x) b h k=x.logics (logicAddress c x ⟨b.val,h.val⟩ k) := by
  have hc := lt_of_lt_of_le hk (hostCounts_covered c x b)
  simp [logitAt,hc,logit,hostInputs,logicAddress,Nat.add_assoc]

theorem host_value [Inhabited A] (c : Host.Call) (x : Host.Input A)
    (b : Fin c.batch) (h : Fin c.heads) (k : Nat) (hk : k < x.counts b.val) (d : Fin c.dimensions) :
    valueAt (hostInputs c x) b h k d=x.values (valueAddress c x ⟨b.val,h.val⟩ k d.val) := by
  have hc := lt_of_lt_of_le hk (hostCounts_covered c x b)
  simp [valueAt,hc,value,hostInputs,valueAddress,pageOffset,Nat.add_mul,Nat.mul_assoc,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem host_normalizable (c : Host.Call) (x : Host.Input (WithBot ℝ))
    (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (valid : NormalizableInputs c x logics values) : Normalizable (hostInputs c x) := by
  constructor
  · intro b h
    obtain ⟨k,r,hk,finite⟩ := valid.positive ⟨b.val,h.val⟩ ⟨b.isLt,h.isLt⟩
    refine ⟨k,hk,?_⟩
    rw [host_logit c x b h k hk,valid.logics _ ⟨b.isLt,h.isLt⟩ k hk,finite]
    exact WithBot.coe_ne_bot
  · intro b h k hk d
    rw [host_value c x b h k hk d,
      valid.values ⟨b.val,h.val⟩ ⟨b.isLt,h.isLt⟩ k hk d.val d.isLt]
    exact WithBot.coe_ne_bot

#axiomsClean hostCounts_covered
#axiomsClean hostInputs_binding
#axiomsClean host_normalizable
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
