import bench.tritonbench_g.rms_matmul_rbe.Proofs.Store

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem maskedScatter_memory {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s t : State α) (h : s.mem=t.mem) :
    (l.foldl (fun acc i => if active i then acc.write r (off i) (val i) else acc) s).mem=
    (l.foldl (fun acc i => if active i then acc.write r (off i) (val i) else acc) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases ha : active i
    · simp only [if_pos ha,State.write,h]
    · simpa only [if_neg ha] using h

theorem storing_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (outputFP8 : Bool) (p : Arguments) (s : State α) (pb pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (storingExecution a outputFP8 p s pb pm pn accumulator hb hm hn ha).val.mem=
      (tileWrite a outputFP8 p pb pm pn accumulator s).mem := by
  cases outputFP8
  all_goals
    simp only [storingExecution,tileWrite,outputActive,outputOffset,outputCell,
      kernelCast,binary,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ofFloat,toFloat]
    apply maskedScatter_memory
    rfl

#axiomsClean maskedScatter_memory
#axiomsClean storing_values
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
