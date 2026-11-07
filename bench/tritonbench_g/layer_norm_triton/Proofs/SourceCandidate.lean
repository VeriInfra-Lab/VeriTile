import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardGraph

/-! A genuine control-flow candidate: expand each admitted single-iteration
forward range into its index assignment followed by the complete body. No
floating expression, cast, reduction, mask, pointer or store is changed. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceCandidate
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- The loop iterator is retained in the final register state. -/
def unrollStmt : ComputeStmt → List ComputeStmt
  | .forRange idx 0 _ _ body => [.assign .nat [] idx (.alg (.constNat 0))]++body
  | st => [st]

def admissible : ComputeStmt → Prop
  | .forRange _ start stop stride _ => start=0 ∧ 0<stop ∧ stop≤stride
  | _ => True

def unroll (k : ComputeKernel) : ComputeKernel :=
  .mk k.inputs k.outputs (k.surfaceBody.flatMap unrollStmt)

theorem step_eq [Inhabited α] (m : Algebra α) (st : ComputeStmt)
    (h : admissible st) (s : State α) : run m (unrollStmt st) s=step m st s := by
  cases st <;> try simp [unrollStmt,run,bind]
  case forRange idx start stop stride body =>
    obtain ⟨rfl,hs,hb⟩ := h
    simp only [unrollStmt,List.cons_append,List.nil_append,run,step,
      evalExpr,evalOp_unfold,bind,Option.bind,pure]
    exact (range_one m idx stop stride hs hb body s).symm

theorem run_eq [Inhabited α] (m : Algebra α) (code : List ComputeStmt)
    (h : ∀ st∈code,admissible st) (s : State α) :
    run m (code.flatMap unrollStmt) s=run m code s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    rw [List.flatMap_cons,run_append,step_eq m st (h st (by simp)),run]
    cases he : step m st s with
    | none => rfl
    | some t => simpa only [Option.bind_some] using ih (fun st hs => h st (by simp [hs])) t

def forward (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  TypedSource.forwardCandidateKernel "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps

/-- The complete Python loop-expanded candidate executes with its own register
state. Only its actual full memory is compared with the original. -/
theorem candidate_executes [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ) (s : State α) :
    ∃ t,FP.Structural.exec (engine m) (forward stride N B eps) s=some t := by
  simp [FP.Structural.exec,forward,TypedSource.forwardCandidateKernel,forwardCandidateKernel,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,integralMod,bop,State.setReg,FP.Structural.store,
    ComputeDType.eraseDType,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]

theorem candidate_memory [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (s t : State α) (h : FP.Structural.exec (engine m) (forward stride N B eps) s=some t) :
    t.mem=(SourceForwardGraph.final m stride N B eps s).mem := by
  simp [FP.Structural.exec,forward,TypedSource.forwardCandidateKernel,forwardCandidateKernel,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,integralMod,bop,State.setReg,FP.Structural.store,
    ComputeDType.eraseDType,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat] at h
  subst t
  simp only [show (0.0 : ℝ)=0 by norm_num,show (1.0 : ℝ)=1 by norm_num]
  simp only [SourceForwardGraph.final,SourceForwardGraph.row,SourceForwardGraph.vector,
    SourceForwardGraph.output,SourceForwardGraph.inverse,SourceForwardGraph.variance,
    SourceForwardGraph.centered,SourceForwardGraph.mean,SourceForwardGraph.sum,
    SourceForwardGraph.xload,SourceForwardGraph.lit,SourceForwardGraph.bin,SourceForwardGraph.cast]
  apply SourceForwardGraph.fold_memory_congr
  · dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write]
    rfl
  · intro a b i hab
    dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
    by_cases hi : i.1.val<N <;> simp [hi,State.write,hab,nat_product_address] <;> rfl

theorem forward_observations [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (hn : 0<N) (hb : N≤B) (s : State α) :
    ∃ original candidate,
      FP.Structural.exec (engine m) (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s=some original ∧
      FP.Structural.exec (engine m) (forward stride N B eps) s=some candidate ∧
      original.mem=candidate.mem := by
  obtain ⟨original,ho⟩ := SourceForward.source_executes m stride N B eps hn hb s
  obtain ⟨candidate,hc⟩ := candidate_executes m stride N B eps s
  exact ⟨original,candidate,ho,hc,
    (SourceForwardGraph.source_memory m stride N B eps hn hb s original ho).trans
      (candidate_memory m stride N B eps s candidate hc).symm⟩

/-- With explicit signed source literal typing, the loop-expanded program
also preserves every final register. This is proved from both full sources. -/
theorem forward_equiv [Inhabited α] (m : Algebra α) (stride N B : Nat) (eps : ℝ)
    (hn : 0<N) (hb : N≤B) (s : State α) :
    FP.Structural.exec m (forward stride N B eps) s=
      FP.Structural.exec m (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s := by
  simp [FP.Structural.exec,forward,TypedSource.forwardCandidateKernel,forwardCandidateKernel,
    TypedSource.forward,forwardKernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,signed_range_one _ _ N B hn hb,numeric,natLt,numericLt,numericLe,numericEq,integralMod,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,TileShape.axisDim,
    TileShape.eraseAxis,TileShape.insertAxisIndex,Region.cast,Option.bind,ofFloat,toFloat]

#axiomsClean forward_equiv
#axiomsClean forward_observations
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceCandidate
