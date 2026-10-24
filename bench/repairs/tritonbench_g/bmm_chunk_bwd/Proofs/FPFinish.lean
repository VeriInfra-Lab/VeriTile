import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPCarry

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPFinish
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault FPReference
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

instance (s : State α) (BM BN : Nat) (p : RealProgram.Parameters) :
    DecidablePred (active s BM BN p) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def write (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [BM,BN]).foldl (fun s i => if active initial BM BN p i then
    s.write p.Db (offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i)
      (.mk .fp32 (expected M D BM BN BCS p initial i)) else s) initial

theorem masked_mem_congr {ι : Type} (r : RegionName) (off : ι → Nat)
    (v w : ι → Cell α) (a : ι → Prop) [DecidablePred a] (l : List ι)
    (s t : State α) (hm : s.mem = t.mem) (hv : ∀ i, a i → v i = w i) :
    (l.foldl (fun x i => if a i then x.write r (off i) (v i) else x) s).mem =
      (l.foldl (fun x i => if a i then x.write r (off i) (w i) else x) t).mem := by
  induction l generalizing s t with
  | nil => exact hm
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : a i
    · simp only [if_pos hi]
      apply ih
      simp [State.write,hm,hv i hi]
    · simpa only [if_neg hi] using ih s t hm

theorem finish (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial s : State α)
    (h : FPCarry.Holds M D BM BN BCS p initial
      ((length initial p+BCS-1)/BCS) ((length initial p+BCS-1)/BCS) s) :
    ∃ final, run (engine (backend M D)) (parts BM BN BCS p).after s = some final ∧
      final.mem = (write M D BM BN BCS p initial).mem := by
  have hb : s.regs .nat [] "pid_b" = some (fun _ => initial.pids 1) := by
    simpa [setup] using h.live.fixed .nat [] "pid_b" (by decide)
  have hc : s.regs .nat [] "pid_c" = some (fun _ => chunk initial p) := by
    simpa [setup,chunk] using h.live.fixed .nat [] "pid_c" (by decide)
  have hh : s.regs .nat [] "pid_h" = some (fun _ => head initial p) := by
    simpa [setup,head,chunk,bop,Broadcast.leftIndex,Broadcast.rightIndex] using h.live.fixed .nat [] "pid_h" (by decide)
  have hm : s.regs .nat [] "pid_m" = some (fun _ => initial.pids 0/((p.K+BN-1)/BN)) := by
    simpa [setup] using h.live.fixed .nat [] "pid_m" (by decide)
  have hn : s.regs .nat [] "pid_n" = some (fun _ => initial.pids 0%((p.K+BN-1)/BN)) := by
    simpa [setup] using h.live.fixed .nat [] "pid_n" (by decide)
  have hl : s.regs .nat [] "chunk_size_limit" = some (fun _ => length initial p) := by
    have he := h.live.fixed .nat [] "chunk_size_limit" (by decide)
    by_cases hc : p.chunk < p.seqlen-initial.pids 2/p.groups*p.chunk
    · simpa [setup,length,chunk,hc,Nat.min_eq_left (Nat.le_of_lt hc)] using he
    · simpa [setup,length,chunk,hc,Nat.min_eq_right (Nat.le_of_not_gt hc)] using he
  obtain ⟨final,hfinal⟩ := finish_success M D BM BN BCS p initial s h.live
  refine ⟨final,hfinal,?_⟩
  have hz := h.acc
  cases hr : p.hasResidual
  all_goals
    simp only [parts, source, RealProgram.source, TypedSource.specialize,
      ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
      beq_self_eq_true, ↓reduceIte] at hfinal
    simp (config := {maxSteps := 100000}) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, backend, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop,
      hb,hc,hh,hm,hn,hl,hz,hr,State.setReg,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,FP.Structural.store] at hfinal
    rw [←hfinal]
    unfold write
    apply masked_mem_congr
    · exact h.live.mem
    · intro i hi
      simp [expected,hr,active,offset,row,column,hi,h.live.mem]

#axiomsClean masked_mem_congr
#axiomsClean finish
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPFinish
