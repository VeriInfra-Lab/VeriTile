import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealGridResult

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def modelOutputLane (c : Call) : (o : Fin 3) → TileIndex (outputShape c o) → Lane c.BM c.D
  | 0=>fun j=>.center j.1
  | 1=>fun j=>.denominator j.1
  | 2=>fun j=>.output j.1 j.2.1

def modelFormula (c : Call) (v : Inputs c) : Outputs c := fun o j=>
  let point:=(outputCoordinates c o).symm j
  (modelLaneValue c v point.1 (modelOutputLane c o point.2)).unbotD 0

theorem model_output_location (c : Call) (o : Fin 3) (j : OutputPoint c o) :
    address c j.1 (modelOutputLane c o j.2)=(outputRegion o,outputAddress c o j) := by
  fin_cases o <;> simp [modelOutputLane,address,outputRegion,outputAddress,Call.io,RealContract.io,
    RealIO.IO.outputWindowFor,Call.params,Call.carryAddress,Call.row,Call.outAddress,
    Geometry.Strides.at,Nat.add_assoc]

theorem model_job_initial (c : Call) (idx : GridIndex c.grid) (s : State Real.R) :
    (modelJob c idx).initial s=Reference.Composition.prepare c.grid idx s := rfl

theorem model_schedule_run (c : Call) (jobs : List (GridIndex c.grid)) (s : State Real.R) :
    runSchedule (jobs.map (modelJob c)) s=
      Reference.Composition.schedule Real.ops.algebra (Reference.launched c) c.grid jobs s := by
  induction jobs generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    simp only [List.map_cons,runSchedule,Reference.Composition.schedule,Invocation.run,model_job_initial]
    change (exec _ _ _).bind _=(exec _ _ _).bind _
    congr 1
    funext t
    exact ih t

theorem model_schedule_safe (c : Call) (positive : 0<c.BN) (v : Inputs c) (bounds : RegionName → Nat)
    (bounded : (modelCore c).InBounds bounds (packInputs c v)) (jobs : List (GridIndex c.grid))
    (s : State Real.R) : ScheduleSafe bounds (jobs.map (modelJob c)) s := by
  induction jobs generalizing s with
  | nil=>trivial
  | cons idx rest ih=>
    refine ⟨?_,fun t _=>ih t⟩
    change TraceSafe _ bounds _ ((modelJob c idx).initial s)
    rw [model_job_initial]
    exact SourceSafety.source_safe c Real.ops positive bounds _ (model_windows c v bounds bounded idx s)

theorem model_output_cell (c : Call) (valid : Valid c) (v : Inputs c) (s : State Real.R)
    (binding : (modelCore c).InputBinding (packInputs c v) s) (o : Fin 3) (j : OutputPoint c o)
    (live : outputActive c o) :
    (Reference.gridResult c Real.ops s).mem (outputRegion o) (outputAddress c o j)=
      .mk .real (modelLaneValue c v j.1 (modelOutputLane c o j.2)) := by
  have distinct : (Reference.Composition.allPrograms c.grid).Nodup := Finset.nodup_toList _
  have h:=model_schedule_outputs c valid v _ distinct s
    (fun idx _ i k active=>model_input_pin c v s binding i (idx,k) active)
    j.1 (Reference.Composition.allPrograms_complete c.grid j.1) (modelOutputLane c o j.2)
    (by fin_cases o <;> exact live)
  rw [model_output_location] at h
  exact h

theorem model_frame (c : Call) (v : Inputs c) (s : State Real.R) :
    (modelCore c).Frame (packInputs c v) s (Reference.gridResult c Real.ops s) := by
  intro rg off outside
  apply ArrayLocality.schedule_frame
  intro idx _ lane active equal
  have avoid (o : Fin 3) (j : OutputPoint c o) (live : outputActive c o) :
      (outputRegion o,outputAddress c o j)≠(rg,off) := by
    intro same
    have h:=outside.1 o (NativeInterface.outputEncoded c o j) live
    change rg≠(modelOutputPorts c)[o].buf ∨ off≠outputAddress c o
      (NativeInterface.outputPoint c o (NativeInterface.outputEncoded c o j)) at h
    rw [NativeInterface.outputPoint_encoded] at h
    have region : (modelOutputPorts c)[o].buf=outputRegion o := by fin_cases o <;> rfl
    rw [region] at h
    exact h.elim (fun ne=>ne (congrArg Prod.fst same).symm) (fun ne=>ne (congrArg Prod.snd same).symm)
  cases lane with
  | center r=>exact avoid 0 (idx,r,()) trivial ((model_output_location c 0 (idx,r,())).symm.trans equal)
  | denominator r=>exact avoid 1 (idx,r,()) active ((model_output_location c 1 (idx,r,())).symm.trans equal)
  | output r d=>exact avoid 2 (idx,r,d,()) trivial ((model_output_location c 2 (idx,r,d,())).symm.trans equal)

theorem model_implements (c : Call) (valid : Valid c) : (modelIO c).Implements (modelFormula c) := by
  apply ProgramIO.Implements.intro
  · intro raw bounds bounded s binding
    have h:=model_schedule_safe c valid.blockN (unpackInputs c raw) bounds
    have roundtrip : packInputs c (unpackInputs c raw)=raw := (modelInputsEquiv c).symm_apply_apply raw
    rw [roundtrip] at h
    exact h bounded _ s
  · intro raw s binding
    let v:=unpackInputs c raw
    have roundtrip : packInputs c v=raw := (modelInputsEquiv c).symm_apply_apply raw
    have pinned : (modelCore c).InputBinding (packInputs c v) s := by rw [roundtrip]; exact binding
    obtain ⟨final,ran,memory⟩:=Reference.complete_grid c Real.ops valid.blockN s
    refine ⟨final,?_,?_,?_⟩
    · exact (model_schedule_run c _ s).trans ran
    · intro o j active
      change outputActive c o at active
      have cell:=model_output_cell c valid v s pinned o (NativeInterface.outputPoint c o j) active
      rw [←memory] at cell
      fin_cases o
      all_goals
        simpa only [modelIO,LogicalProgramIO.lift,modelOutputsEquiv,modelInputsEquiv,
          modelCore,modelFormula,packOutputs,mapCell,mapValue,cell,
          modelOutputPorts,NativeInterface.outputPorts,List.ofFn_succ,List.getElem_cons_zero,
          List.getElem_cons_succ,cast_eq] using congrArg (mapCell (fun x : Real.R=>x.unbotD 0)) cell
    · intro rg off outside
      rw [memory]
      have h:=model_frame c v s rg off
      rw [roundtrip] at h
      exact h outside

#axiomsClean model_implements
#specNonCircular modelFormula avoiding [Reference.result,Reference.iterate,Reference.gridResult,FP.Structural.exec,FP.Structural.run,Kernels.source]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
