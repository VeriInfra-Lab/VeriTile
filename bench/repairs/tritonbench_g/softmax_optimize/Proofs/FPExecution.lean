import bench.repairs.tritonbench_g.softmax_optimize.Proofs.TypedSource
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelRewrite
import VeriTile.Triton.Float.StructuralIO

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The source uses equality only for its two explicit -infinity guards. -/
def withEq (A : Algebra α) (eq : α → α → Bool) : Algebra α :=
  { A with compareEq := fun _ _ => some eq }

def previous (N B : Nat) : Int := ((N:Int)+B-1)/(B:Int)*B-B

def Ready (shape : TileShape) (N B pid : Nat) (s : State α) : Prop :=
  s.regs .int [] "pid_m"=some (fun _ => (pid:Int)) ∧
  s.regs .int [] "cols"=some (fun _ => (N:Int)) ∧
  s.regs .int [] "prev_multiple"=some (fun _ => previous N B) ∧
  (s.regs .real shape "m").isSome ∧ (s.regs .real shape "z").isSome

def Frame (Y : RegionName) (s t : State α) : Prop :=
  ∀ r i,r≠Y → t.mem r i=s.mem r i

theorem prelude (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B : Nat) (hB : 0<B) (s : State α) :
    ∃ t, run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq)) ((TypedSource.body Y X N B).take 5) s=some t ∧
      Ready [B] N B (s.pids 0) t ∧ t.mem=s.mem := by
  have nonneg : ¬((N:Int)+(B:Int)<1) := by omega
  simp (config := {maxSteps := 300000}) [TypedSource.body,kernel,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,withEq,
    ComputeDType.eraseDType,numeric,numericEq,numericLt,numericLe,bop,ofFloat,toFloat,
    State.setReg,integralFloorDiv,integralMod,Ready,previous,nonneg,TileShape.axisDim,hB]
  rfl

theorem input_body (A : Algebra α) (eq : α → α → Bool)
    (X : RegionName) (N B pid : Nat) (tail : Bool) (s : State α)
    (ready : Ready [B] N B pid s) (index : Int) :
    ∃ t, run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      (if tail then TypedSource.tailInput X N B else TypedSource.fullInput X N B)
      (s.setReg "start_n" .int [] (fun _ => index))=some t ∧
      Ready [B] N B pid t ∧ t.mem=s.mem := by
  rcases ready with ⟨hp,hc,hprev,hm,hz⟩
  obtain ⟨m,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨z,hz⟩ := Option.isSome_iff_exists.mp hz
  cases tail <;>
    simp (config := {maxSteps := 300000}) [TypedSource.fullInput,TypedSource.tailInput,
      TypedSource.loopBody,TypedSource.body,kernel,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,withEq,
      ComputeDType.eraseDType,numeric,numericEq,numericLt,numericLe,bop,ofFloat,toFloat,
      State.setReg,integralFloorDiv,integralMod,Ready,Region.cast,hp,hc,hprev,hm,hz]

theorem reduction (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (hB : 0<B) (s : State α)
    (ready : Ready [B] N B pid s) :
    ∃ t, run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq)) (((TypedSource.body Y X N B).drop 7).take 4) s=some t ∧
      Ready [] N B pid t ∧ t.mem=s.mem := by
  have nonneg : ¬((N:Int)+(B:Int)<1) := by omega
  rcases ready with ⟨hp,hc,hprev,hm,hz⟩
  obtain ⟨m,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨z,hz⟩ := Option.isSome_iff_exists.mp hz
  simp (config := {maxSteps := 300000}) [TypedSource.body,kernel,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,withEq,
    ComputeDType.eraseDType,numeric,numericEq,numericLt,numericLe,bop,ofFloat,toFloat,
    State.setReg,integralFloorDiv,integralMod,Ready,previous,hp,hc,hprev,hm,hz,nonneg,TileShape.axisDim,hB]
  rfl

@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (xs : List ι) (s : State α) :
    (xs.foldl (fun st i => if active i then st.write r (off i) (val i) else st) s).regs = s.regs := by
  rw [State.masked_scatter_eq_filter, State.scatter_regs]

theorem output_body (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (tail : Bool) (s : State α)
    (ready : Ready [] N B pid s) (index : Int) :
    ∃ t, run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      (if tail then TypedSource.tailOutput Y X N B else TypedSource.fullOutput Y X N B)
      (s.setReg "start_n" .int [] (fun _ => index))=some t ∧
      Ready [] N B pid t ∧ Frame Y s t := by
  rcases ready with ⟨hp,hc,hprev,hm,hz⟩
  obtain ⟨m,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨z,hz⟩ := Option.isSome_iff_exists.mp hz
  cases tail <;>
    simp (config := {maxSteps := 300000}) [TypedSource.fullOutput,TypedSource.tailOutput,
      TypedSource.loopBody,TypedSource.body,kernel,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
      VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,withEq,
      ComputeDType.eraseDType,numeric,numericEq,numericLt,numericLe,bop,ofFloat,toFloat,
      State.setReg,integralFloorDiv,integralMod,Ready,Region.cast,Frame,hp,hc,hprev,hm,hz]
  · intro r i h
    exact State.scatter_frame _ _ _ _ r i (Or.inl h) _
  · intro r i h
    exact State.masked_scatter_frame _ _ _ _ _ r i (Or.inl h) _

#axiomsClean prelude
#axiomsClean input_body
#axiomsClean reduction
#axiomsClean output_body
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
