import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Store
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.LoopFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem maskedWrite_regs {α ι : Type} (region : ι → RegionName) (offset : ι → Nat)
    (value : ι → Cell α) (active : ι → Bool) (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s).regs=
      s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih]
    cases active i <;> rfl

theorem maskedWrite_pids {α ι : Type} (region : ι → RegionName) (offset : ι → Nat)
    (value : ι → Cell α) (active : ι → Bool) (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s).pids=
      s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih]
    cases active i <;> rfl

theorem storing_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (outputFP8 : Bool) (p : Arguments) (s : State α) (pb pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    HeaderPreserved s
      (storingExecution a outputFP8 p s pb pm pn accumulator hb hm hn ha).val := by
  intro d sh name member
  simp [headerNames] at member
  rcases member with rfl|rfl|rfl|rfl|rfl
  all_goals cases outputFP8 <;> simp [storingExecution,maskedWrite_regs]

theorem storing_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (outputFP8 : Bool) (p : Arguments) (s : State α) (pb pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (storingExecution a outputFP8 p s pb pm pn accumulator hb hm hn ha).val.pids=s.pids := by
  cases outputFP8 <;> simp [storingExecution,maskedWrite_pids]

#axiomsClean storing_header
#axiomsClean storing_pids
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
