import bench.tritonbench_g.attention_fwd_triton3.IO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEGrid
import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSESafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeImplementation
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.CSEInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership NativeInterface
noncomputable section
set_option maxHeartbeats 3000000

/-- CSE is inside the key loop; the sole runtime scale slot and its raw-word
interpretation retain the original complete-quotation binding. -/
theorem only_scale_slot (c : Call) :
    (Kernels.keyMaskCSE c).surfaceBody=
      (Kernels.keyMaskCSE (ScaleBinding.template c)).surfaceBody.take 17 ++
      [.assign .real [] "qk_scale" (.alg (.const c.scale))] ++
      (Kernels.keyMaskCSE (ScaleBinding.template c)).surfaceBody.drop 18 := rfl

theorem job_initial (c : Call) (idx : GridIndex c.grid) (s : State Native.Expr) :
    (job c idx).initial s=Reference.Composition.prepare c.grid idx s := rfl

theorem schedule_run (c : Call) (jobs : List (GridIndex c.grid)) (s : State Native.Expr) :
    runSchedule (jobs.map (job c)) s=
      Reference.Composition.schedule Native.backend.algebra (KeyMaskCSE.launched (ScaleBinding.template c))
        c.grid jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    simp only [List.map_cons,runSchedule,Reference.Composition.schedule,Invocation.run,job_initial]
    change (exec _ _ _).bind _=(exec _ _ _).bind _
    congr 1
    funext t
    exact ih t

theorem schedule_safe (c : Call) (positive : 0<c.BN) (v : Inputs) (bounds : RegionName → Nat)
    (bounded : (programIO c).InBounds bounds v) (jobs : List (GridIndex c.grid))
    (s : State Native.Expr) : ScheduleSafe bounds (jobs.map (job c)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons idx rest ih =>
    refine ⟨?_,fun t _ => ih t⟩
    change TraceSafe _ bounds _ ((job c idx).initial s)
    rw [job_initial]
    exact KeyMaskCSE.source_safe (ScaleBinding.template c) Native.backend positive bounds _
      (windows c v bounds bounded idx s)

theorem implements (c : Call) (positive : 0<c.BN) : (programIO c).Implements (expressionFormula c) := by
  apply ArrayProgramIO.ImplementsOn.intro
  · intro v _ bounds bounded s _
    exact schedule_safe c positive v bounds bounded _ s
  · intro v _ s binding
    obtain ⟨final,ran,memory⟩ := KeyMaskCSE.complete_grid (ScaleBinding.template c) Native.backend positive s
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run c _ s).trans ran
    · intro o j active
      change RealInterface.outputActive c o at active
      have cell := output_cell c v s binding o (outputPoint c o j) active
      rw [←memory] at cell
      fin_cases o
      all_goals
        simpa only [programIO,NativeInterface.programIO,expressionFormula,mapCell,mapValue,id_eq,cell,outputPorts,
          List.ofFn_succ,List.getElem_cons_zero,List.getElem_cons_succ,cast_eq] using
            congrArg (mapCell (id : Native.Expr → Native.Expr)) cell
    · intro rg off outside
      rw [memory]
      exact frame c v s rg off outside


#axiomsClean schedule_run
#axiomsClean schedule_safe
#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.CSEInterface
