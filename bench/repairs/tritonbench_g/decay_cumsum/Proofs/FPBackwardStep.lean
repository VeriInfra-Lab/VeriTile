import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPStep

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackwardStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps
open VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep (assign_compute)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def direct (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (u : State α)
    (r : RegionName) : Values α .real [h.BK] := fun i => M.fp32Load (raw h.BK r off u i)
def qterm (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.mul M (direct M h off u "dqo" i) (FPReference.exp2 M (g i))
def kterm (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g last : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.mul M (direct M h off u "dko" i) (FPReference.exp2 M (FPReference.sub M (last i) (g i)))
def qvalue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.add M (direct M h off u "dqi" i) (qterm M h off g u i)
def kvalue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g last : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.add M (direct M h off u "dki" i) (kterm M h off g last u i)
def delta (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g last : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] := fun i =>
  FPReference.sub M (FPReference.mul M (qvalue M h off g u i) (direct M h off u "q" i))
    (FPReference.mul M (kvalue M h off g last u i) (direct M h off u "k" i))
def nextCarry (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g last acc : Values α .real [h.BK])
    (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.add M (acc i) (delta M h off g last u i)

def result (M : Algebra α) (h : Call) (off : Fin h.BK → Nat) (g last acc : Values α .real [h.BK])
    (u : State α) : State α :=
  let v := u.setReg "_dq1" .real [h.BK] (direct M h off u "dqi")
  let v := v.setReg "_dq2" .real [h.BK] (direct M h off u "dqo")
  let v := v.setReg "_dq2" .real [h.BK] (qterm M h off g u)
  let v := v.setReg "_dq" .real [h.BK] (qvalue M h off g u)
  let v := directScatter M h.BK "dqo" off (qvalue M h off g u) v
  let v := v.setReg "_dk1" .real [h.BK] (direct M h off u "dki")
  let v := v.setReg "_dk2" .real [h.BK] (direct M h off u "dko")
  let v := v.setReg "_dk2" .real [h.BK] (kterm M h off g last u)
  let v := v.setReg "_dk" .real [h.BK] (kvalue M h off g last u)
  let v := directScatter M h.BK "dko" off (kvalue M h off g last u) v
  let v := v.setReg "_q" .real [h.BK] (direct M h off u "q")
  let v := v.setReg "_k" .real [h.BK] (direct M h off u "k")
  let v := v.setReg "_dg" .real [h.BK] (delta M h off g last u)
  let v := v.setReg "cum_grad_dg" .real [h.BK] (nextCarry M h off g last acc u)
  let v := scatter M h.BK "dg" off (nextCarry M h off g last acc u) v
  let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" (fun i => off i-h.width))
  let v := v.setReg "p_k" .ptr [h.BK] (ptr h.BK "k" (fun i => off i-h.width))
  let v := v.setReg "p_q" .ptr [h.BK] (ptr h.BK "q" (fun i => off i-h.width))
  let v := v.setReg "p_dq_inner" .ptr [h.BK] (ptr h.BK "dqi" (fun i => off i-h.width))
  let v := v.setReg "p_dk_inner" .ptr [h.BK] (ptr h.BK "dki" (fun i => off i-h.width))
  let v := v.setReg "p_dq_inter" .ptr [h.BK] (ptr h.BK "dqo" (fun i => off i-h.width))
  let v := v.setReg "p_dk_inter" .ptr [h.BK] (ptr h.BK "dko" (fun i => off i-h.width))
  v.setReg "p_dg" .ptr [h.BK] (ptr h.BK "dg" (fun i => off i-h.width))

theorem executes (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (g last acc : Values α .real [h.BK]) (u : State α)
    (hq : u.regs .ptr [h.BK] "p_q"=some (ptr h.BK "q" off))
    (hk : u.regs .ptr [h.BK] "p_k"=some (ptr h.BK "k" off))
    (hg : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" off))
    (hdqi : u.regs .ptr [h.BK] "p_dq_inner"=some (ptr h.BK "dqi" off))
    (hdki : u.regs .ptr [h.BK] "p_dk_inner"=some (ptr h.BK "dki" off))
    (hdqo : u.regs .ptr [h.BK] "p_dq_inter"=some (ptr h.BK "dqo" off))
    (hdko : u.regs .ptr [h.BK] "p_dk_inter"=some (ptr h.BK "dko" off))
    (hdg : u.regs .ptr [h.BK] "p_dg"=some (ptr h.BK "dg" off))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true))
    (hgate : u.regs .real [h.BK] "_g"=some g)
    (hlast : u.regs .real [h.BK] "last_g"=some last)
    (ha : u.regs .real [h.BK] "cum_grad_dg"=some acc) :
    run (engine M) (backwardArithmetic h) u=some (result M h off g last acc u) := by
  unfold backwardArithmetic assign32
  rw [run,data_load M h.BK "dqi" off "_dq1" "p_dq_inner" _ hdqi hm]
  simp only [bind,Option.bind_some]
  rw [run,data_load M h.BK "dqo" off "_dq2" "p_dq_inter" _ (by simp [hdqo]) (by simp [hm])]
  simp only [bind,Option.bind_some]
  rw [assign_compute _ _ _ _ _ _ _ (gated M h.BK "_dq2" "_g" (direct M h off u "dqo") g _ (by simp [direct,raw]; rfl) (by simp [hgate]))]
  rw [assign_compute _ _ _ _ _ _ _ (add_eval M h.BK "_dq1" "_dq2" (direct M h off u "dqi") (qterm M h off g u) _ (by simp [direct,raw]; rfl) (by simp [qterm]; rfl))]
  rw [run,direct_store_eval M h.BK "dqo" off "p_dq_inter" "_dq" (qvalue M h off g u) _ (by simp [hdqo]) (by simp [hm]) (by simp [qvalue]; rfl)]
  simp only [bind,Option.bind_some]
  rw [run,data_load M h.BK "dki" off "_dk1" "p_dk_inner" _ (by simp [hdki]) (by simp [hm])]
  simp only [bind,Option.bind_some]
  rw [run,data_load M h.BK "dko" off "_dk2" "p_dk_inter" _ (by simp [hdko]) (by simp [hm])]
  simp only [bind,Option.bind_some]
  rw [assign_compute _ _ _ _ _ _ _ (difference_gated M h.BK "_dk2" "last_g" "_g" (direct M h off u "dko") last g _ (by simp [direct,raw]; rfl) (by simp [hlast]) (by simp [hgate]))]
  rw [assign_compute _ _ _ _ _ _ _ (add_eval M h.BK "_dk1" "_dk2" (direct M h off u "dki") (kterm M h off g last u) _ (by simp [direct,raw]; rfl) (by simp [kterm]; rfl))]
  rw [run,direct_store_eval M h.BK "dko" off "p_dk_inter" "_dk" (kvalue M h off g last u) _ (by simp [hdko]) (by simp [hm]) (by simp [kvalue]; rfl)]
  simp only [bind,Option.bind_some]
  rw [run,data_load M h.BK "q" off "_q" "p_q" _ (by simp [hq]) (by simp [hm])]
  simp only [bind,Option.bind_some]
  rw [run,data_load M h.BK "k" off "_k" "p_k" _ (by simp [hk]) (by simp [hm])]
  simp only [bind,Option.bind_some]
  rw [assign_compute _ _ _ _ _ _ _ (gradient_eval M h.BK (qvalue M h off g u) (kvalue M h off g last u) (direct M h off u "q") (direct M h off u "k") _ (by simp [qvalue]; rfl) (by simp [kvalue]; rfl) (by simp [direct,raw]; rfl) (by simp [direct,raw]; rfl))]
  rw [assign_compute _ _ _ _ _ _ _ (add_eval M h.BK "cum_grad_dg" "_dg" acc (delta M h off g last u) _ (by simp [ha]) (by simp [delta]; rfl))]
  rw [run,store_eval M h.BK "dg" off "p_dg" "cum_grad_dg" (nextCarry M h off g last acc u) _ (by simp [hdg]) (by simp [hm]) (by simp [nextCarry]; rfl)]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "g" off "p_g" _ (by simp [hg])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "k" off "p_k" _ (by simp [hk])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "q" off "p_q" _ (by simp [hq])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "dqi" off "p_dq_inner" _ (by simp [hdqi])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "dki" off "p_dk_inner" _ (by simp [hdki])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "dqo" off "p_dq_inter" _ (by simp [hdqo])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "dko" off "p_dk_inter" _ (by simp [hdko])]
  simp only [bind,Option.bind_some]
  rw [run,retreat_eval M h.BK h.width "dg" off "p_dg" _ (by simp [hdg])]
  simp only [bind,Option.bind_some]
  simp [run,result,direct,raw,qterm,kterm,qvalue,kvalue,delta,nextCarry]
  rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackwardStep
