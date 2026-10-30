import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.FPPhysical

namespace VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The integer-only candidate has exactly the same complete symbolic state,
including failures and unspecified inactive weight payloads. -/
theorem candidate_exec (p : Parameters) (s : State Term) :
    FP.Structural.exec algebra (TensorSource.commutedSource p) s=
      FP.Structural.exec algebra (TensorSource.source p) s := by
  let original := (TensorSource.source p).surfaceBody
  let candidate := (TensorSource.commutedSource p).surfaceBody
  have prefixRun : run algebra (candidate.take 3) s=run algebra (original.take 3) s := by
    simp [candidate,original,TensorSource.source,TensorSource.commutedSource,
      indexCommutedKernel,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,numeric,bop,State.setReg,Option.bind,Nat.mul_comm]
    delta bop
    simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  change run algebra (candidate.take 3 ++ original.drop 3) s=run algebra (original.take 3 ++ original.drop 3) s
  rw [run_append,run_append,prefixRun]

def candidateRowRun (p : Parameters) (b m : Nat) (u : Nat → Term) (before : Memory Term) : Option (Memory Term) :=
  (FP.Structural.exec algebra (TensorSource.commutedSource p) (state p before u b m)).map (result p before)

def candidateGrid (p : Parameters) (u : Nat → Term) : List (Nat × Nat) → Memory Term → Option (Memory Term)
  | [],before=>some before
  | (b,m)::rest,before=>(candidateRowRun p b m u before).bind (candidateGrid p u rest)

theorem candidate_grid (p : Parameters) (u : Nat → Term) (order : List (Nat × Nat)) (before : Memory Term) :
    candidateGrid p u order before=grid p u order before := by
  induction order generalizing before with
  | nil=>rfl
  | cons bm rest ih=>
    rcases bm with ⟨b,m⟩
    simp only [candidateGrid,grid,candidateRowRun,rowRun,candidate_exec]
    apply congrArg (Option.bind _)
    funext memory
    exact ih memory

/-- Native target from actual logical tensor payloads. Source code, machine
states, memory updates, and interpreter execution do not occur in this target. -/
def nativeTensorResult (Q : Numerics Torch.FP32.Word) (profile : Profile) (p : Parameters)
    (x w : Torch.FP32.Tensor) (before : Memory Torch.FP32.Word) (b m k : Nat) : Torch.FP32.Word :=
  observe Q profile (logical p.width blockSize p.eps
    (fun j=>.input (tensorRead before x (flatIndex p b m j)))
    (fun j=>.input (tensorRead before w j)) k)

/-- Observation is applied only after a successful full typed grid. -/
def observedGrid (Q : Numerics Torch.FP32.Word) (profile : Profile) (p : Parameters)
    (u : Nat → Torch.FP32.Word) (order : List (Nat × Nat)) (before : Memory Torch.FP32.Word) :
    Option (Memory Torch.FP32.Word) :=
  (grid p (fun k=>.input (u k)) order (entryGraph before)).map (projectMemory Q profile)

def observedCandidateGrid (Q : Numerics Torch.FP32.Word) (profile : Profile) (p : Parameters)
    (u : Nat → Torch.FP32.Word) (order : List (Nat × Nat)) (before : Memory Torch.FP32.Word) :
    Option (Memory Torch.FP32.Word) :=
  (candidateGrid p (fun k=>.input (u k)) order (entryGraph before)).map (projectMemory Q profile)

theorem tensor_result (Q : Numerics Torch.FP32.Word) (profile : Profile) (p : Parameters)
    (views : TensorBinding x w out p base) (sep : Separation p) (unique : OutputUnique p)
    (u : Nat → Torch.FP32.Word) (schedule : Schedule p) (before : Memory Torch.FP32.Word) :
    ∃ symbolicFinal final,
      grid p (fun k=>.input (u k)) schedule.order (entryGraph before)=some symbolicFinal ∧
      candidateGrid p (fun k=>.input (u k)) schedule.order (entryGraph before)=some symbolicFinal ∧
      observedGrid Q profile p u schedule.order before=some final ∧
      observedCandidateGrid Q profile p u schedule.order before=some final ∧
      final=projectMemory Q profile symbolicFinal ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        tensorRead final views.output (flatIndex p b m k)=
          nativeTensorResult Q profile p views.input views.weight before b m k) ∧
      final=referenceMemory p (nativeTensorResult Q profile p views.input views.weight before) before ∧
      (∀ a o,Outside p a o → final a o=before a o) := by
  let final := gridModel p (fun k=>Term.input (u k)) schedule.order (entryGraph before)
  obtain ⟨values,frame⟩ := grid_result p sep unique (fun k=>Term.input (u k)) schedule (entryGraph before)
  have projected : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
      projectMemory Q profile final p.output.allocation (outputOffset p b m k)=
        nativeTensorResult Q profile p views.input views.weight before b m k := by
    intro b m k hb hm hk
    unfold projectMemory
    rw [show final p.output.allocation (outputOffset p b m k)=rowLogical p (entryGraph before) b m k from
      values b m k hb hm hk]
    apply congrArg (observe Q profile)
    exact logical_congr p.width blockSize p.eps _ _ _ _
      (fun j hj=>congrArg Term.input (views.input_read before b m j hb hm hj).symm)
      (fun j hj=>congrArg Term.input (views.weight_read before j hj).symm) k hk
  have outside : ∀ a o,Outside p a o → projectMemory Q profile final a o=before a o := by
    intro a o h
    unfold projectMemory
    rw [show final a o=entryGraph before a o from frame a o h]
    rfl
  refine ⟨final,projectMemory Q profile final,grid_run _ _ _ _,?_,?_,?_,rfl,?_,?_,outside⟩
  · rw [candidate_grid]
    exact grid_run _ _ _ _
  · simp only [observedGrid,grid_run,Option.map_some]
    rfl
  · simp only [observedCandidateGrid,candidate_grid,grid_run,Option.map_some]
    rfl
  · intro b m k hb hm hk
    exact (views.output_read _ b m k hb hm hk).trans (projected b m k hb hm hk)
  · exact memory_ext p _ before _ projected outside

#axiomsClean candidate_exec
#axiomsClean tensor_result
#guard_msgs (drop info) in
#specNonCircular nativeTensorResult avoiding [kernel,code,graph,rowModel,gridModel,grid,rowRun,state]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
