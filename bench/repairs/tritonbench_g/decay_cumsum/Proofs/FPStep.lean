import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPOps

namespace VeriTile.Bench.Repairs.DecayCumsum.FPStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps
open VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep (assign_compute)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def fwdValue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (acc : Values α .real [h.BK]) (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.add M (acc i)
    (FPReference.mul M (FPReference.gate M (raw h.BK "g" off u i))
      (M.literal (some .fp32) .real 1.44269504))

def forwardResult (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (acc : Values α .real [h.BK]) (u : State α) : State α :=
  let value := fwdValue M h off acc u
  let v := u.setReg "_g" .real [h.BK] (fun i => FPReference.gate M (raw h.BK "g" off u i))
  let v := v.setReg "cum_decay" .real [h.BK] value
  let v := scatter M h.BK "go" off value v
  let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" (fun i => off i+h.width))
  v.setReg "p_go" .ptr [h.BK] (ptr h.BK "go" (fun i => off i+h.width))

theorem forward (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (acc : Values α .real [h.BK]) (u : State α)
    (hg : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" off))
    (ho : u.regs .ptr [h.BK] "p_go"=some (ptr h.BK "go" off))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true))
    (ha : u.regs .real [h.BK] "cum_decay"=some acc) :
    run (engine M) (forwardBody h) u=some (forwardResult M h off acc u) := by
  unfold forwardBody assign32
  rw [assign_compute _ _ _ _ _ _ _ (gate_load M h.BK "g" off "p_g" u hg hm)]
  rw [assign_compute _ _ _ _ _ _ _ (forward_update M h.BK acc (fun i => FPReference.gate M (raw h.BK "g" off u i)) _ (by simp [ha]) (by simp))]
  rw [run,store_eval M h.BK "go" off "p_go" "cum_decay" (fwdValue M h off acc u) _ (by simp [ho]) (by simp [hm]) (by simp [fwdValue]; rfl)]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "g" off "p_g" _ (by simp [hg])]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "go" off "p_go" _ (by simp [ho])]
  simp only [bind, Option.bind_some, run]
  rfl

def prepareQValue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (scale : ℝ) (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.prepareQ M scale (raw h.BK "q" off u i) (raw h.BK "g" off u i)
def prepareKValue (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (last : Values α .real [h.BK]) (u : State α) : Values α .real [h.BK] :=
  fun i => FPReference.mul M (M.fp32Load (raw h.BK "k" off u i))
    (FPReference.exp2 M (FPReference.sub M (last i) (FPReference.gate M (raw h.BK "g" off u i))))

def prepareResult (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (scale : ℝ) (last : Values α .real [h.BK]) (u : State α) : State α :=
  let qv := prepareQValue M h off scale u
  let kv := prepareKValue M h off last u
  let v := u.setReg "_q" .real [h.BK] (fun i => M.fp32Load (raw h.BK "q" off u i))
  let v := v.setReg "_k" .real [h.BK] (fun i => M.fp32Load (raw h.BK "k" off u i))
  let v := v.setReg "_g" .real [h.BK] (fun i => FPReference.gate M (raw h.BK "g" off u i))
  let v := v.setReg "_q" .real [h.BK] qv
  let v := v.setReg "_k" .real [h.BK] kv
  let v := scatter M h.BK "kg" off kv v
  let v := scatter M h.BK "qg" off qv v
  let v := v.setReg "p_q" .ptr [h.BK] (ptr h.BK "q" (fun i => off i+h.width))
  let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" (fun i => off i+h.width))
  let v := v.setReg "p_k" .ptr [h.BK] (ptr h.BK "k" (fun i => off i+h.width))
  let v := v.setReg "p_kg" .ptr [h.BK] (ptr h.BK "kg" (fun i => off i+h.width))
  v.setReg "p_qg" .ptr [h.BK] (ptr h.BK "qg" (fun i => off i+h.width))

theorem prepare (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (scale : ℝ) (last : Values α .real [h.BK]) (u : State α)
    (hq : u.regs .ptr [h.BK] "p_q"=some (ptr h.BK "q" off))
    (hk : u.regs .ptr [h.BK] "p_k"=some (ptr h.BK "k" off))
    (hg : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" off))
    (hqo : u.regs .ptr [h.BK] "p_qg"=some (ptr h.BK "qg" off))
    (hko : u.regs .ptr [h.BK] "p_kg"=some (ptr h.BK "kg" off))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true))
    (hl : u.regs .real [h.BK] "last_decay"=some last) :
    run (engine M) (prepareBody h scale) u=some (prepareResult M h off scale last u) := by
  unfold prepareBody assign32
  rw [run,data_load M h.BK "q" off "_q" "p_q" _ hq hm]
  try simp only [bind, Option.bind_some]
  rw [run,data_load M h.BK "k" off "_k" "p_k" _ (by simp [hk]) (by simp [hm])]
  try simp only [bind, Option.bind_some]
  rw [assign_compute _ _ _ _ _ _ _ (gate_load M h.BK "g" off "p_g" _ (by simp [hg]) (by simp [hm]))]
  rw [assign_compute _ _ _ _ _ _ _ (prepare_q M h.BK scale (fun i => M.fp32Load (raw h.BK "q" off u i)) (fun i => FPReference.gate M (raw h.BK "g" off u i)) _ (by simp [raw]) (by simp [raw]))]
  rw [assign_compute _ _ _ _ _ _ _ (difference_gated M h.BK "_k" "last_decay" "_g" (fun i => M.fp32Load (raw h.BK "k" off u i)) last (fun i => FPReference.gate M (raw h.BK "g" off u i)) _ (by simp [raw]) (by simp [hl]) (by simp [raw]))]
  rw [run,store_eval M h.BK "kg" off "p_kg" "_k" (prepareKValue M h off last u) _ (by simp [hko]) (by simp [hm]) (by simp [prepareKValue,raw]; rfl)]
  try simp only [bind, Option.bind_some]
  rw [run,store_eval M h.BK "qg" off "p_qg" "_q" (prepareQValue M h off scale u) _ (by simp [hqo]) (by simp [hm]) (by simp [prepareQValue,FPReference.prepareQ,raw]; rfl)]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "q" off "p_q" _ (by simp [hq])]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "g" off "p_g" _ (by simp [hg])]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "k" off "p_k" _ (by simp [hk])]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "kg" off "p_kg" _ (by simp [hko])]
  try simp only [bind, Option.bind_some]
  rw [run,advance_eval M h.BK h.width "qg" off "p_qg" _ (by simp [hqo])]
  simp only [bind, Option.bind_some, run]
  rfl

#axiomsClean forward
#axiomsClean prepare
end
end VeriTile.Bench.Repairs.DecayCumsum.FPStep
