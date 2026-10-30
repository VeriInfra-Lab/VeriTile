import bench.repairs.tritonbench_g.chunk_linear_attn.Candidate
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyOutputComplete

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def parts (p : Parameters) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (forwardOHoisted p scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def headStmt (p : Parameters) : ComputeStmt :=
  .assign .nat [] "qk_head" (.alg (.mul .nat .nil
    (.ref .nat [] "i_bh") (.constNat (p.T*p.K))))

theorem before (p : Parameters) (scale : ℝ) :
    (parts p scale).before = (outputParts p scale).before ++ [headStmt p] := rfl

theorem after (p : Parameters) (scale : ℝ) :
    (parts p scale).after = (outputParts p scale).after := rfl

theorem decomposition (p : Parameters) (scale : ℝ) :
    (forwardOHoisted p scale).surfaceBody = (parts p scale).before ++
      [.forRangeDyn "i_k" (parts p scale).start (parts p scale).stop
        (parts p scale).stride (parts p scale).body] ++ (parts p scale).after := rfl

def Head (p : Parameters) (pid : PID) (s : State α) : Prop :=
  s.regs .nat [] "qk_head" = some (fun _ => pid.z*(p.T*p.K))

def entry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :=
  (outputEntry M D p scale s).val.setReg "qk_head" .nat [] (fun _ => s.pids 2*(p.T*p.K))

theorem entry_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    run (engine (withDot M D)) (parts p scale).before s = some (entry M D p scale s) := by
  rw [before,run_append,(outputEntry M D p scale s).property]
  have bh := (outputEntry_ready M D p scale s).1.bh
  simp [headStmt,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,bh,entry]; rfl

theorem entry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    OutputReady p ⟨s.pids 0,s.pids 1,s.pids 2⟩ s.mem (entry M D p scale s) ∧
    HasOutput p (outputInitial M p) (entry M D p scale s) ∧
    Head p ⟨s.pids 0,s.pids 1,s.pids 2⟩ (entry M D p scale s) := by
  have ready := outputEntry_ready M D p scale s
  refine ⟨?_,?_,?_⟩
  · constructor
    · exact ready.1.memory
    · simpa [entry,State.setReg] using ready.1.iv
    · simpa [entry,State.setReg] using ready.1.it
    · simpa [entry,State.setReg] using ready.1.bh
    · simpa [entry,State.setReg] using ready.1.mask
  · simpa [HasOutput,entry,State.setReg] using ready.2
  · simp [Head,entry,State.setReg]

theorem body_suffix (p : Parameters) (scale : ℝ) :
    (parts p scale).body.drop 2=(outputParts p scale).body.drop 2 := rfl

theorem first_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (kb : Nat)
    (ctx : OutputReady p pid input s) (head : Head p pid s) :
    run (engine (withDot M D)) ((parts p scale).body.take 2) (s.setReg "i_k" .nat [] (fun _ => kb)) =
      run (engine (withDot M D)) ((outputParts p scale).body.take 2) (s.setReg "i_k" .nat [] (fun _ => kb)) := by
  have hh : s.regs .nat [] "qk_head"=some (fun _ => pid.z*(p.T*p.K)) := head
  simp [parts,forwardOHoisted,outputParts,forwardO,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.bh,ctx.iv,ctx.it,hh,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]

theorem body_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (kb : Nat)
    (ctx : OutputReady p pid input s) (head : Head p pid s) :
    run (engine (withDot M D)) (parts p scale).body (s.setReg "i_k" .nat [] (fun _ => kb)) =
      run (engine (withDot M D)) (outputParts p scale).body (s.setReg "i_k" .nat [] (fun _ => kb)) := by
  rw [←List.take_append_drop 2 (parts p scale).body,run_append,
    first_run M D p scale pid input s kb ctx head,body_suffix,
    ←run_append,List.take_append_drop]


private def KeepsHead : ComputeStmt → Prop
  | .assign _ _ name _ => name ≠ "qk_head"
  | _ => False

private theorem step_head (M : Algebra α) (stmt : ComputeStmt) (valid : KeepsHead stmt)
    (s t : State α) (ran : step M stmt s=some t) :
    t.regs .nat [] "qk_head"=s.regs .nat [] "qk_head" := by
  cases stmt <;> simp only [KeepsHead] at valid
  rename_i dtype shape name rhs
  cases he : evalExpr M rhs s with
  | none => simp [step,he] at ran
  | some v =>
    have same : s.setReg name dtype shape v=t := by simpa [step,he] using ran
    subst t
    simp [State.setReg,Ne.symm valid]

private theorem run_head (M : Algebra α) (code : List ComputeStmt)
    (valid : ∀ stmt∈code,KeepsHead stmt) (s t : State α) (ran : run M code s=some t) :
    t.regs .nat [] "qk_head"=s.regs .nat [] "qk_head" := by
  induction code generalizing s with
  | nil => simpa [run] using congrArg (fun r => r.map (fun u => u.regs .nat [] "qk_head")) ran.symm
  | cons stmt rest ih =>
    cases he : step M stmt s with
    | none => simp [run,he] at ran
    | some u =>
      have tail : run M rest u=some t := by simpa [run,he] using ran
      exact (ih (fun st member => valid st (by simp [member])) u tail).trans
        (step_head M stmt (valid stmt (by simp)) s u he)

theorem body_head (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (kb : Nat) (a : OutputAcc p α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p a s) (head : Head p pid s) :
    Head p pid (outputBody M D p scale pid input s kb a ctx ha).val := by
  have valid : ∀ stmt∈(outputParts p scale).body,KeepsHead stmt := by
    simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,KeepsHead]
  have preserved := run_head (engine (withDot M D)) (outputParts p scale).body valid
    (s.setReg "i_k" .nat [] (fun _ => kb))
    (outputBody M D p scale pid input s kb a ctx ha).val
    (outputBody M D p scale pid input s kb a ctx ha).property
  unfold Head
  rw [preserved]
  simpa [State.setReg] using head

#axiomsClean body_run

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
