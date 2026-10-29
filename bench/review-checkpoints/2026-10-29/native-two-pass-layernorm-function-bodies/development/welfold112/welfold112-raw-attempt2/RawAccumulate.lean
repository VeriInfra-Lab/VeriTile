import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawSyntax
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Bounds

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Raw
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Actual source registers. Pass one does not pretend a mean register exists. -/
structure Ready {α : Type} (phase : Bool) (s t : State α) {B : Nat}
    (values : Fin B → α) (mean : α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  values : t.regs .real [1,B] (if phase then "_tmp12" else "_tmp3") =
    some (fun i => values i.2.1)
  mean : phase=Bool.true → t.regs .real [1,1] "tmp6"=some (fun _ => mean)

variable {α : Type} [Inhabited α]

theorem prefix_run (M : Algebra α) (s : State α) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) :
    ∃ t,run M (initBody S D B tree) s=some t ∧
      Ready Bool.false s t (fun _ : Fin B => zero M) (zero M) := by
  simp [initBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat]
  constructor <;> simp [State.setReg,zero,Fin.eq_zero,FP.Structural.bop]
  all_goals first | rfl | (funext i; simp [FP.Structural.bop])

theorem phase_run (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B k : Nat) (tree : DSL.WelfordSource.Tree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean) :
    ∃ final,run M (phaseBody phase X S D B tree)
        (t.setReg "roffset" .nat [] (fun _ => k*B))=some final ∧
      Ready phase s final (advance M phase s X D B k mean values) mean := by
  have hm := h.mean
  cases phase <;>
    simp [phaseBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,numericLt,
      Region.cast,ofFloat,toFloat,h.values,h.row,h.base,h.mem,hm]
  all_goals constructor <;> try simp [State.setReg,advance,loaded,zero,FP.Structural.bop,
    ofFloat,toFloat,Fin.eq_zero,h.mem,h.pids,h.undef,h.row,h.base,h.mean]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [FP.Structural.bop])
  all_goals try dsimp [FP.Structural.bop]
  all_goals try simp_all
  all_goals norm_num [FP.Structural.bop]
  all_goals rename_i i hi
  all_goals simp only [if_neg (Nat.not_lt_of_ge hi)]

/-- The recurrence is indexed by chunk number; the actual source iterator is
its exact element offset. The count is derived from the Python range bounds. -/
theorem phase_loop (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B start count : Nat) (tree : DSL.WelfordSource.Tree B)
    (hB : 0<B) (count_eq : start+count=blockCount D B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean) :
    ∃ final,range M "roffset" (start*B) D B (phaseBody phase X S D B tree) t=some final ∧
      Ready phase s final (recurrence M phase s X D B mean start count values) mean := by
  induction count generalizing start t values with
  | zero =>
      have stopped : ¬start*B<D := by
        intro active
        have := (range_offset_iff D B start hB).mpr active
        omega
      exact ⟨t,by rw [range,if_neg (Nat.ne_of_gt hB),if_neg stopped],h⟩
  | succ count ih =>
      have active : start*B<D := (range_offset_iff D B start hB).mp (by omega)
      obtain ⟨u,hu,hp⟩ := phase_run M phase s t X S D B start tree values mean h
      obtain ⟨v,hv,hq⟩ := ih u (start+1) (by omega) _ hp
      refine ⟨v,?_,hq⟩
      rw [range,if_neg (Nat.ne_of_gt hB),if_pos active,hu]
      simpa only [Option.bind_some,show start*B+B=(start+1)*B by simp [Nat.add_mul]] using hv

#axiomsClean prefix_run
#axiomsClean phase_run
#axiomsClean phase_loop
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Raw
