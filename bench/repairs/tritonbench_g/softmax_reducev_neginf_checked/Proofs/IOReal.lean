import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.IO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOGrid
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.RealSource

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def RealDomain {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs (WithBot ℝ) c counts) : Prop := LengthsMatch xs ∧ Normalizable xs

theorem real_source_inputs (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job)
    (xs : Inputs (WithBot ℝ) c counts) (s : State (WithBot ℝ))
    (bound : (realIO c counts work).InputBinding xs s) (domain : RealDomain xs)
    (b : Fin c.batch) (h : Fin c.heads) :
    RealModel.Inputs (Host.parameters c)
      ((invocation RealModel.model c ⟨b.val,h.val⟩).initial s)
      (logitAt xs b h) (fun k d => (valueAt xs b h k d).unbotD 0) := by
  have len := source_length (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
    xs s bound domain.1 b h
  constructor
  · intro k hk
    have hk' : k < counts b := by simpa [len] using hk
    have value := source_logit (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
      xs s bound b h k (0 : Fin 64) (by simpa using hk')
    simpa only [RealModel.keyLogicOffset,logicOffset,Fin.val_zero,Nat.add_zero,id_eq] using value
  · intro k hk d
    have hk' : k < counts b := by simpa [len] using hk
    have value := source_value (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
      xs s bound domain.1 b h k (0 : Fin 64) d (by simpa using hk')
    have finite := domain.2.2 b h k hk' d
    have coe : valueAt xs b h k d=↑((valueAt xs b h k d).unbotD 0) := by
      obtain ⟨r,hr⟩ := WithBot.ne_bot_iff_exists.mp finite
      rw [←hr]
      rfl
    apply Eq.trans (b := valueAt xs b h k d) _ coe
    simpa only [RealModel.keyValueOffset,valueOffset,location,active,Fin.val_zero,
      Nat.add_zero,if_pos hk,id_eq] using value

theorem row_mathematical (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job)
    (xs : Inputs (WithBot ℝ) c counts) (s : State (WithBot ℝ))
    (bound : (realIO c counts work).InputBinding xs s) (domain : RealDomain xs)
    (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) :
    rawRow (fun _ => RealModel.model) id c counts xs b h d=
      ↑(RealModel.target (counts b) (logitAt xs b h) (fun k => (valueAt xs b h k d).unbotD 0)) := by
  have len := source_length (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
    xs s bound domain.1 b h
  obtain ⟨k,hk,_⟩ := domain.2.1 b h
  have positive : 0 < counts b := lt_of_le_of_lt (Nat.zero_le k) hk
  have result := RealModel.source_value (Host.parameters c)
    ((invocation RealModel.model c ⟨b.val,h.val⟩).initial s)
    (logitAt xs b h) (fun k d => (valueAt xs b h k d).unbotD 0)
    (real_source_inputs c counts work xs s bound domain b h)
    (by change 0<64; decide) (by rw [len]; exact positive) (d,())
  rw [row_logical (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work
    xs s bound domain.1,len] at result
  exact result

theorem real_implements (c : Host.Call) (counts : Fin c.batch → Nat)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    (realIO c counts work).ImplementsOn RealDomain (mathematical c counts) := by
  intro xs domain bounds hb s bound
  obtain ⟨safe,t,run,out,frame⟩ := ordered_implements (fun _ => RealModel.model) id c
    (fun _ _ _ => id) counts work complete xs domain.1 bounds hb s bound
  refine ⟨safe,t,run,?_,frame⟩
  intro port i active
  have hp : port=(0 : Fin 1) := by
    apply Fin.ext
    change port.val=0
    have : port.val<1 := port.isLt
    omega
  subst port
  have result := out (0 : Fin 1) i active
  change mapCell id (t.mem "Out" (Host.address c ⟨i.1.val,i.2.1.val⟩ i.2.2.1.val))=
    .mk .fp32 (↑(RealModel.target (counts i.1) (logitAt xs i.1 i.2.1)
      (fun k => (valueAt xs i.1 i.2.1 k i.2.2.1).unbotD 0)))
  change mapCell id (t.mem "Out" (Host.address c ⟨i.1.val,i.2.1.val⟩ i.2.2.1.val))=
    .mk .fp32 (rawRow (fun _ => RealModel.model) id c counts xs i.1 i.2.1 i.2.2.1) at result
  rw [row_mathematical c counts work xs s bound domain] at result
  exact result

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
