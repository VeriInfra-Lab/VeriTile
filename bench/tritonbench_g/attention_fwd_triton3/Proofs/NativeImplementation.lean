import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeBindings
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeFrame
import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxHeartbeats 3000000

theorem job_initial (c : Call) (idx : GridIndex c.grid) (s : State Native.Expr) :
    (job c idx).initial s=Reference.Composition.prepare c.grid idx s := rfl

theorem schedule_run (c : Call) (jobs : List (GridIndex c.grid)) (s : State Native.Expr) :
    runSchedule (jobs.map (job c)) s=
      Reference.Composition.schedule Native.backend.algebra (Reference.launched (ScaleBinding.template c))
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
    exact SourceSafety.source_safe (ScaleBinding.template c) Native.backend positive bounds _
      (windows c v bounds bounded idx s)

theorem output_location (c : Call) (o : Fin 3) (j : RealInterface.OutputPoint c o) :
    Ownership.address c j.1 (outputLane c o j.2)=
      (RealInterface.outputRegion o,RealInterface.outputAddress c o j) := by
  fin_cases o <;> simp [outputLane,Ownership.address,RealInterface.outputRegion,
    RealInterface.outputAddress,Call.io,RealContract.io,RealIO.IO.outputWindowFor,
    Call.params,Call.carryAddress,Call.row,Call.outAddress,Geometry.Strides.at,Nat.add_assoc]

theorem output_active (c : Call) (o : Fin 3) (j : RealInterface.OutputPoint c o)
    (active : RealInterface.outputActive c o) : Ownership.active c (outputLane c o j.2) := by
  fin_cases o <;> exact active


theorem template_active (c : Call) (lane : Lane c.BM c.D) :
    Ownership.active (ScaleBinding.template c) lane ↔ Ownership.active c lane := by
  cases lane <;> rfl

theorem template_address (c : Call) (idx : GridIndex c.grid) (lane : Lane c.BM c.D) :
    Ownership.address (ScaleBinding.template c) idx lane=Ownership.address c idx lane := by
  cases lane <;> rfl

theorem output_cell (c : Call) (v : Inputs) (s : State Native.Expr)
    (binding : (programIO c).InputBinding v s) (o : Fin 3) (j : RealInterface.OutputPoint c o)
    (active : RealInterface.outputActive c o) :
    (Reference.gridResult (ScaleBinding.template c) Native.backend s).mem
      (RealInterface.outputRegion o) (RealInterface.outputAddress c o j)=
        .mk .real (expressionAt c v o j) := by
  have loc := output_location c o j
  have live := output_active c o j active
  have hit := ArrayLocality.schedule_real_hit (ScaleBinding.template c) Native.backend
    (Reference.Composition.allPrograms c.grid) s j.1
    (Reference.Composition.allPrograms_complete c.grid j.1) (outputLane c o j.2)
    ((template_active c _).mpr live)
  rw [template_address] at hit
  change ArrayLocality.IsReal ((Reference.gridResult (ScaleBinding.template c) Native.backend s).mem
    (Ownership.address c j.1 (outputLane c o j.2)).1
    (Ownership.address c j.1 (outputLane c o j.2)).2) at hit
  rw [loc] at hit
  obtain ⟨value,cell⟩ := hit
  have same := ArrayLocality.schedule_same_reads (ScaleBinding.template c) Native.backend
    (Reference.Composition.allPrograms c.grid) s (arrayState v) (input_binding c v s binding)
  have member : RealInterface.outputRegion o∈ArrayLocality.buffers := by
    fin_cases o <;> simp [RealInterface.outputRegion,ArrayLocality.buffers]
  have read := same (RealInterface.outputRegion o) member (RealInterface.outputAddress c o j)
  change ((Reference.gridResult (ScaleBinding.template c) Native.backend s).mem _ _).read .real=
    expressionAt c v o j at read
  rw [cell,Cell.read_same] at read
  exact cell.trans (congrArg (Cell.mk .real) read)

theorem frame (c : Call) (v : Inputs) (s : State Native.Expr) :
    (programIO c).Frame v s (Reference.gridResult (ScaleBinding.template c) Native.backend s) := by
  intro rg off outside
  apply ArrayLocality.schedule_frame
  intro idx _ lane active equal
  rw [template_address] at equal
  rw [template_active] at active
  have avoid (o : Fin 3) (j : RealInterface.OutputPoint c o) (live : RealInterface.outputActive c o) :
      (RealInterface.outputRegion o,RealInterface.outputAddress c o j)≠(rg,off) := by
    intro same
    have h := outside.1 o (outputEncoded c o j) live
    change rg≠(outputPorts c)[o].buf ∨ off≠RealInterface.outputAddress c o
      (outputPoint c o (outputEncoded c o j)) at h
    rw [outputPoint_encoded] at h
    have region : (outputPorts c)[o].buf=RealInterface.outputRegion o := by fin_cases o <;> rfl
    rw [region] at h
    exact h.elim (fun ne => ne (congrArg Prod.fst same).symm)
      (fun ne => ne (congrArg Prod.snd same).symm)
  cases lane with
  | center r =>
    exact avoid 0 (idx,r,PUnit.unit) trivial
      ((output_location c 0 (idx,r,PUnit.unit)).symm.trans equal)
  | denominator r =>
    exact avoid 1 (idx,r,PUnit.unit) active
      ((output_location c 1 (idx,r,PUnit.unit)).symm.trans equal)
  | output r d =>
    exact avoid 2 (idx,r,d,PUnit.unit) trivial
      ((output_location c 2 (idx,r,d,PUnit.unit)).symm.trans equal)

theorem implements (c : Call) (positive : 0<c.BN) : (programIO c).Implements (expressionFormula c) := by
  apply ArrayProgramIO.ImplementsOn.intro
  · intro v _ bounds bounded s _
    exact schedule_safe c positive v bounds bounded _ s
  · intro v _ s binding
    obtain ⟨final,ran,memory⟩ := Reference.complete_grid (ScaleBinding.template c) Native.backend positive s
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run c _ s).trans ran
    · intro o j active
      change RealInterface.outputActive c o at active
      have cell := output_cell c v s binding o (outputPoint c o j) active
      rw [←memory] at cell
      fin_cases o
      all_goals
        simpa only [programIO,expressionFormula,mapCell,mapValue,id_eq,cell,outputPorts,
          List.ofFn_succ,List.getElem_cons_zero,List.getElem_cons_succ,cast_eq] using
            congrArg (mapCell (id : Native.Expr → Native.Expr)) cell
    · intro rg off outside
      rw [memory]
      exact frame c v s rg off outside

theorem observes (c : Call) (shape : Native.Shape c) (schedule : Raw.Schedule c)
    (actual reference : Raw.Ops) (laws : Raw.LocalLaws actual reference) (scale : Native.Word) :
    (programIO c).Observes (observer c shape schedule actual scale) (formula c shape schedule reference scale) := by
  rw [Raw.operations_eq actual reference laws]
  have h := ArrayProgramIO.Observes.of_implements (implements c shape.blockN)
    (observer c shape schedule reference scale)
  convert h using 1
  funext v o j
  fin_cases o <;> rfl

#axiomsClean implements
#axiomsClean observes
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
