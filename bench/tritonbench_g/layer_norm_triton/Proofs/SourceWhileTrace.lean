import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGrouped
import VeriTile.Triton.Float.SignedAffineWhileInvariant
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceWhileTrace
open VeriTile.Triton FP.Structural SourceModel
open FP.Structural.AffineWhile
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def Control (M : Nat) (current : Int) (s : State α) : Prop :=
  NormalizedSignedCounter s "row" current ∧ s.regs .int [] "M"=some (fun _ => (M : Int))

theorem prelude_control (m : Model α) (stride M N G B : Nat) (s t : State α)
    (ran : run (engine m) (SourceGrouped.prelude stride M N G B) s=some t) :
    Control M (s.pids 0 : Int) t := by
  simp [SourceGrouped.prelude,SourceGrouped.code,TypedSource.groupedBackwardKernel,groupedBackwardKernel,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,State.setReg,
    ComputeDType.eraseDType,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat] at ran
  subst t
  constructor
  · simp only [NormalizedSignedCounter,State.setReg]
    congr 1
    funext dtype shape name
    by_cases h : name="row" <;> simp [h]
  · simp [State.setReg]

theorem body_control (m : Model α) (stride M N G B : Nat) (current : Int) (s t : State α)
    (ready : SourceGrouped.Ready B s) (control : Control M current s)
    (ran : run (engine m) (SourceGrouped.body stride M N G B) s=some t) :
    Control M (current+G) t := by
  obtain ⟨normalized,boundValue⟩ := control
  have counterValue := normalized_signed_read s "row" current normalized
  obtain ⟨⟨group,hgroup⟩,⟨cols,hcols⟩,⟨mask,hmask⟩,⟨dwp,hdwp⟩,⟨dbp,hdbp⟩,
    ⟨dw,hdw⟩,⟨db,hdb⟩,⟨row,hrow⟩⟩ := ready
  by_cases same : current=group PUnit.unit <;>
    simp [SourceGrouped.body,SourceGrouped.code,TypedSource.groupedBackwardKernel,groupedBackwardKernel,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      engine,numeric,natLt,numericLt,numericLe,numericEq,bop,State.setReg,FP.Structural.store,
      ComputeDType.eraseDType,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb,counterValue,same] at ran
  all_goals subst t
  all_goals constructor
  all_goals first
    | (simp only [NormalizedSignedCounter,State.setReg]
       congr 1
       funext dtype shape name
       by_cases h : name="row" <;> simp [h,bop,Broadcast.leftIndex,Broadcast.rightIndex,same])
    | simp [State.setReg,boundValue]
  all_goals split_ifs <;> subst_vars <;> rfl

theorem execution (m : Model α) (stride M N G B : Nat) (positive : 0<G)
    (initial first final : State α)
    (prelude : run (engine m) (SourceGrouped.prelude stride M N G B) initial=some first)
    (loop : rangeInt (engine m) "row" (initial.pids 0 : Int) (M : Int) G
      (SourceGrouped.body stride M N G B) first=some final) :
    Trace (engine m) (signedGuard "row" "M") (SourceGrouped.body stride M N G B) first final := by
  obtain ⟨other,otherRan,ready⟩ := SourceGrouped.prelude_success m stride M N G B initial
  have eq : other=first := Option.some.inj (otherRan.symm.trans prelude)
  subst other
  let I (current : Int) (s : State α) : Prop := Control M current s ∧ SourceGrouped.Ready B s
  apply signed_range_to_trace_invariant (engine m) "row" "M" G (M : Int) positive
    (SourceGrouped.body stride M N G B) I
  · intro current s invariant
    exact invariant.1
  · intro current s t invariant ran
    have normalized := invariant.1.1
    obtain ⟨next,nextRan,nextReady⟩ := SourceGrouped.body_success m stride M N G B current s invariant.2
    rw [normalized] at nextRan
    have eq : next=t := Option.some.inj (nextRan.symm.trans ran)
    subst next
    exact ⟨body_control m stride M N G B current s t invariant.2 invariant.1 ran,nextReady⟩
  · exact ⟨prelude_control m stride M N G B initial first prelude,ready⟩
  · exact loop
/-- The complete source executes through its prefix, an ordinary while trace,
and suffix. No successful loop execution is an external premise. -/
theorem source_execution (m : Model α) (stride M N G B : Nat) (positive : 0<G)
    (initial : State α) :
    ∃ first final output,
      run (engine m) (SourceGrouped.prelude stride M N G B) initial=some first ∧
      Trace (engine m) (signedGuard "row" "M") (SourceGrouped.body stride M N G B) first final ∧
      run (engine m) (SourceGrouped.suffix stride M N G B) final=some output ∧
      FP.Structural.exec (engine m)
        (TypedSource.groupedBackwardKernel "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) initial=some output := by
  obtain ⟨first,hfirst,ready⟩ := SourceGrouped.prelude_success m stride M N G B initial
  have control := prelude_control m stride M N G B initial first hfirst
  have hrow := normalized_signed_read first "row" (initial.pids 0 : Int) control.1
  obtain ⟨final,hloop,finalReady⟩ := SourceFinal.range_success (engine m) "row"
    (initial.pids 0 : Int) (M : Int) G (SourceGrouped.body stride M N G B)
    (SourceGrouped.Ready B) (SourceGrouped.body_success m stride M N G B) first ready
  obtain ⟨output,houtput⟩ := SourceGrouped.suffix_success m stride M N G B final finalReady
  refine ⟨first,final,output,hfirst,execution m stride M N G B positive initial first final hfirst hloop,houtput,?_⟩
  change run (engine m) (SourceGrouped.code stride M N G B) initial=some output
  rw [SourceGrouped.code_split,run_append,run_append,hfirst]
  simp only [run,step,evalOp_unfold,bind,Option.bind,hrow]
  change rangeInt (engine m) "row" (initial.pids 0 : Int) (Int.ofNat M) G
    (SourceGrouped.body stride M N G B) first=some final at hloop
  rw [hloop]
  exact houtput
#axiomsClean source_execution
#axiomsClean execution
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceWhileTrace
