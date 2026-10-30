import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelScalar
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Source

/-! Reuse of the complete original execution proof for the actual scale-first
source. Reversing the helper algebra's multiplication arguments is a proof
device: its original dt*A executes exactly the candidate's A*dt under the
unchanged natural graph algebra. The candidate keeps every actual operand in
its source order; numerical comparison is proved separately. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSource
open VeriTile.Triton VeriTile.Triton.FP.Structural
open Kernel Kernel.Scalar Kernel.Math
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def reverseMul (M : Algebra α) : Algebra α :=
  { M with binary := fun precision dtype op a b =>
      match op with
      | .mul => M.binary precision dtype .mul b a
      | _ => M.binary precision dtype op a b }

def reversePrimitives (c : Geometry) (p : Primitives α c) : Primitives α c :=
  { p with model := reverseMul p.model
           scan_support := by intro x; simpa only [reverseMul] using p.scan_support x }

def candidateKernel (c : Geometry) (hasBias useSoftplus : Bool) : ComputeKernel :=
  scaleFirst "dt" "rate" "bias" "out" "sum" c.batches c.sequence c.heads c.chunk
    c.dtBatch c.dtTime c.dtHead c.rateHead c.biasHead
    c.layout.outBatch c.layout.outChunk c.layout.outHead c.layout.outColumn
    c.layout.sumBatch c.layout.sumChunk c.layout.sumHead c.layout.sumColumn
    useSoftplus hasBias c.BH c.BC

def before (c : Geometry) (hasBias useSoftplus : Bool) : List ComputeStmt :=
  Source.prefixBefore (kernel c hasBias useSoftplus).surfaceBody "dA"

def after (c : Geometry) (hasBias useSoftplus : Bool) : List ComputeStmt :=
  (kernel c hasBias useSoftplus).surfaceBody.drop ((before c hasBias useSoftplus).length+1)

def updateStmt (c : Geometry) (reverse : Bool) : ComputeStmt :=
  .assign .real [c.BH,c.BC] "dA"
    (.compute (.alg .fp32 (if reverse then
      .mul .real (.consSame (.consL .nil))
        (.expandDim (shape := [c.BH]) 1 (.ref .real [c.BH] "A")) (.ref .real [c.BH,c.BC] "dt")
    else
      .mul .real (.consSame (.consR .nil))
        (.ref .real [c.BH,c.BC] "dt") (.expandDim (shape := [c.BH]) 1 (.ref .real [c.BH] "A")))))

theorem original_body (c : Geometry) (hasBias useSoftplus : Bool) :
    (kernel c hasBias useSoftplus).surfaceBody =
      before c hasBias useSoftplus ++ [updateStmt c Bool.false] ++ after c hasBias useSoftplus := by
  cases hasBias <;> cases useSoftplus <;> rfl

theorem candidate_body (c : Geometry) (hasBias useSoftplus : Bool) :
    (candidateKernel c hasBias useSoftplus).surfaceBody =
      before c hasBias useSoftplus ++ [updateStmt c Bool.true] ++ after c hasBias useSoftplus := by
  cases hasBias <;> cases useSoftplus <;> rfl

theorem run_of_step_eq [Inhabited α] (M N : Algebra α) (code : List ComputeStmt)
    (h : ∀ st ∈ code, ∀ s, FP.Structural.step M st s = FP.Structural.step N st s)
    (s : State α) : run M code s = run N code s := by
  induction code generalizing s with
  | nil => simp [run]
  | cons st rest ih =>
    simp only [run]
    rw [h st (by simp)]
    exact congrArg (fun next => (FP.Structural.step N st s).bind next)
      (funext (fun t => ih (fun q hq => h q (by simp [hq])) t))

theorem before_steps [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) : ∀ st ∈ before c hasBias useSoftplus, ∀ s,
    FP.Structural.step (engine (reverseMul M)) st s = FP.Structural.step (engine M) st s := by
  cases hasBias <;> cases useSoftplus
  all_goals
    simp [before, Source.prefixBefore, kernel, source, ComputeKernel.surfaceBody]
    repeat' apply And.intro
    all_goals intro s
    all_goals simp [run, FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold,
      engine, Algebra.withDefaultPrecision, reverseMul, resolvePrecision, ComputeDType.eraseDType,
      numeric, numericLe, numericLt, natLe, natLt, FP.Structural.bop, FP.Structural.store]

theorem after_steps [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) : ∀ st ∈ after c hasBias useSoftplus, ∀ s,
    FP.Structural.step (engine (reverseMul M)) st s = FP.Structural.step (engine M) st s := by
  cases hasBias <;> cases useSoftplus
  all_goals
    simp [after, before, Source.prefixBefore, kernel, source, ComputeKernel.surfaceBody]
    repeat' apply And.intro
    all_goals intro s
    all_goals simp [run, FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold,
      engine, Algebra.withDefaultPrecision, reverseMul, resolvePrecision, ComputeDType.eraseDType,
      numeric, numericLe, numericLt, natLe, natLt, FP.Structural.bop, FP.Structural.store]

theorem before_run [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (s : State α) :
    run (engine (reverseMul M)) (before c hasBias useSoftplus) s =
      run (engine M) (before c hasBias useSoftplus) s := by
  exact run_of_step_eq _ _ _ (before_steps M c hasBias useSoftplus) s

theorem after_run [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (s : State α) :
    run (engine (reverseMul M)) (after c hasBias useSoftplus) s =
      run (engine M) (after c hasBias useSoftplus) s := by
  exact run_of_step_eq _ _ _ (after_steps M c hasBias useSoftplus) s

theorem update_run [Inhabited α] (M : Algebra α) (c : Geometry) (s : State α) :
    run (engine (reverseMul M)) [updateStmt c Bool.false] s =
      run (engine M) [updateStmt c Bool.true] s := by
  cases hd : s.regs .real [c.BH,c.BC] "dt" <;>
    cases ha : s.regs .real [c.BH] "A" <;>
    simp [updateStmt, run, FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold,
      engine, Algebra.withDefaultPrecision, reverseMul, hd, ha, FP.Structural.bop, numeric,
      resolvePrecision, ComputeDType.eraseDType, Broadcast.leftIndex, Broadcast.rightIndex,
      Option.bind, TileShape.dropInsertedIndex]
  rfl

/-- Exact complete-source connection; it changes only the helper model's
argument order at the one multiplication site and assumes no numeric law. -/
theorem complete_run [Inhabited α] (M : Algebra α) (c : Geometry)
    (hasBias useSoftplus : Bool) (s : State α) :
    exec (engine (reverseMul M)) (kernel c hasBias useSoftplus) s =
      exec (engine M) (candidateKernel c hasBias useSoftplus) s := by
  have hb : run (engine (reverseMul M)) (before c hasBias useSoftplus) =
      run (engine M) (before c hasBias useSoftplus) := funext (before_run M c hasBias useSoftplus)
  have ht : run (engine (reverseMul M)) (after c hasBias useSoftplus) =
      run (engine M) (after c hasBias useSoftplus) := funext (after_run M c hasBias useSoftplus)
  have hu : run (engine (reverseMul M)) [updateStmt c Bool.false] =
      run (engine M) [updateStmt c Bool.true] := funext (update_run M c)
  change run (engine (reverseMul M)) (kernel c hasBias useSoftplus).surfaceBody s =
    run (engine M) (candidateKernel c hasBias useSoftplus).surfaceBody s
  rw [original_body, candidate_body]
  simp only [run_append, hb, ht, hu]

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSource
