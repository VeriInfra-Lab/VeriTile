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
    (hb : s.regs .int [] "pid_batch"=some (fun _ => (pb:Int)))
    (hm : s.regs .int [] "pid_m"=some (fun _ => (pm:Int)))
    (hn : s.regs .int [] "pid_n"=some (fun _ => (pn:Int)))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (storingExecution a outputFP8 p s pb pm pn accumulator hb hm hn ha).val.mem=
      (tileWrite a outputFP8 p pb pm pn accumulator s).mem := by
  have address (i : TileIndex [16,64]) :
      Int.toNat ((pb:Int)*p.sob+((pm:Int)*16+i.1.val)*p.som+
        ((pn:Int)*64+i.2.1.val)*p.son)=outputOffset p pb pm pn i := by
    rfl
  have active (i : TileIndex [16,64]) :
      ((pm:Int)*16+i.1.val<(p.M:Int) ∧ (pn:Int)*64+i.2.1.val<(p.N:Int)) ↔
        outputActive p pm pn i := by
    change (((pm*16+i.1.val:Nat):Int)<(p.M:Int) ∧
      ((pn*64+i.2.1.val:Nat):Int)<(p.N:Int)) ↔ _
    simp only [Int.ofNat_lt,outputActive]
  cases outputFP8
  all_goals
    simp only [storingExecution,tileWrite,outputActive,outputOffset,outputCell,
      kernelCast,binary,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ofFloat,toFloat,address,active,reduceCtorEq,if_false]
    apply maskedScatter_memory
    rfl

#axiomsClean maskedScatter_memory
#axiomsClean storing_values
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
