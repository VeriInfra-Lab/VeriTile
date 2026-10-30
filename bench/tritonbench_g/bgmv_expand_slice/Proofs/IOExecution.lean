import bench.tritonbench_g.bgmv_expand_slice.IO
import bench.tritonbench_g.bgmv_expand_slice.Proofs.KernelRaw
import VeriTile.Triton.Float.ProgramIO

/-! Actual source execution from arbitrary regional cells. Values use exactly
its declared typed reads; unwritten cells retain their original tags. The
original canonical Data execution remains a supporting special case. -/
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
open VeriTile.Triton FP.Structural Direct Direct.FPValues Signed.Program
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

def data (s : State A) : Data A where
  input := fun i => (s.mem "X" i).read .real
  weights := fun i => (s.mem "W" i).read .fp16
  output := fun i => (s.mem "Y" i).read .real
  indices := fun i => (s.mem "Indices" i).read .int
  other := s.mem
  padding := s.undef

def entry (h : Call) (batch row : Nat) (s : State A) : State A where
  mem := s.mem
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then row else if ax=1 then batch else 0
  numPids := fun ax => if ax=0 then h.N else if ax=1 then h.batches else 1
  undef := s.undef

/-- Constructive evaluation of all three actual metadata-prefix statements. -/
def preguard (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row : Nat) (s : State A) : {t : State A //
    run (machine M reduction) (prelude h) (entry h batch row s)=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,code,Surface.source,
    ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,entry,data,State.setReg,Cell.read_same,Region.cast]
  rfl

/-- Every setup statement succeeds for the local primitives, including the
source's explicit fp32-to-half cast and signed metadata-derived pointers. -/
def preloop (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int)) :
    {t : State A // run (machine M reduction) (setup h)
      (preguard M reduction h batch row s).val=some t} := by
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  cases ev
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [setup,active,code,Surface.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    dsimp [preguard]
    simp (config := {maxSteps := 100000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read_same,ofFloat,toFloat,Region.cast,entry,data,metadata]
    try rfl

def firstEntry (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int)) :=
  (preloop M reduction h batch row slot s metadata).val.setReg "n" .int [] (fun _ => 0)

/-- Constructive execution of the actual typed loop body with an opaque local
FP16 reduction primitive. No complete-output or successful-execution premise. -/
def iteration (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int)) :
    {t : State A // run (machine M reduction) (body h)
      (firstEntry M reduction h batch row slot s metadata)=some t} := by
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  cases ev <;> cases adds
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [body,active,code,Surface.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    dsimp [firstEntry,preloop]
    simp (config := {maxSteps := 300000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,FP.Structural.store,
      State.setReg,Cell.read_same,ofFloat,toFloat,TileShape.axisDim,Region.cast,entry,data,metadata]
    try rfl

theorem iteration_memory (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int))
    (positiveN : 0<h.N) :
    (iteration M reduction h batch row slot s metadata).val.mem=
      ((entry h batch row s).write "Y" (batch*h.columns+row+h.slice)
        (.mk .real (result M reduction h batch slot row (data s) ⟨0,by decide⟩))).mem := by
  have metaRead : (s.mem "Indices" batch).read .int=(slot:Int) := metadata
  have length := signed_length h positiveN
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  change ((N:Int)+N-1)/N=1 at length
  have nonzero : N≠0 := by change 0<N at positiveN;omega
  have length_cast : ((↑(N+N):Int)-1)/(N:Int)=1 := by simpa only [Int.natCast_add] using length
  have positive_sum : ¬((N:Int)+N<1) := by change 0<N at positiveN;omega
  have ptrCast : (N:Int)*K*slot+row*K=((N*K*slot+row*K:Nat):Int) := by push_cast; rfl
  have ptrToNat : ((N:Int)*K*slot+row*K).toNat=N*K*slot+row*K := by rw [ptrCast,Int.toNat_natCast]
  cases ev <;> cases adds
  all_goals
    simp (config := {maxSteps := 300000}) [iteration,firstEntry,preloop,preguard,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read_same,ofFloat,toFloat,TileShape.axisDim,Region.cast,entry,data,metaRead,
      Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_ediv,length,length_cast,nonzero,positive_sum,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul,
      result,sums,Direct.FPValues.products,loaded,weight,old,State.masked_scatter_eq_filter,Signed.Readback.active_lanes,Signed.Readback.active_lanes_zero,
      ptrToNat,Int.toNat_natCast]
    try simp only [State.masked_scatter_eq_filter,Signed.Readback.active_lanes_zero,List.foldl_cons,List.foldl_nil]
    simp (config := {maxSteps := 100000}) [State.write,Direct.FPValues.products,loaded,weight,bop,Broadcast.leftIndex,Broadcast.rightIndex,ptrToNat,Nat.mul_comm,length_cast,nonzero,positive_sum,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    rw [Signed.Readback.active_lanes_zero]
    simp only [List.foldl_cons,List.foldl_nil,State.write]
    unfold Direct.FPValues.products FP.Structural.bop
    simp [loaded,weight,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,length_cast,nonzero,positive_sum,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    try rfl

/-- The single source iteration retains the original signed-length law. -/
theorem preloop_length (M : Algebra A) (reduction : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int))
    (positiveN : 0<h.N) :
    (preloop M reduction h batch row slot s metadata).val.regs .int [] "split_n_length"=
      some (fun _ => 1) := by
  have length := signed_length h positiveN
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  have positive_sum : ¬((N:Int)+N<1) := by change 0<N at positiveN;omega
  cases ev
  all_goals
    simp [preloop,integralFloorDiv]
    funext i
    simpa only [if_neg positive_sum] using length

theorem condition_eval (M : Algebra A) (R : Reduction A) (h : Call)
    (batch row : Nat) (s : State A) :
    FP.Structural.evalOp (machine M R) none guard (preguard M R h batch row s).val=
      some (fun _ => decide ((data s).indices batch≠-1)) := by
  apply Signed.Control.integer_guard
  rfl

/-- Complete authoritative signed source on arbitrary entry cells; exactly one
active output is written, or the signed -1 branch preserves all memory. -/
theorem source_executes (M : Algebra A) (R : Reduction A) (h : Call)
    (p : Geometry.Program h) (s : State A)
    (valid : (data s).indices p.batch=-1 ∨ 0≤(data s).indices p.batch) :
    ∃ final,(invocation M R h p.batch p.row).run s=some final ∧
      final.mem=(if (data s).indices p.batch=-1 then s.mem else
        (s.write "Y" (FPGrid.address h p)
          (.mk .real (FPGrid.value M R h p (data s)))).mem) := by
  by_cases sentinel : (data s).indices p.batch=-1
  · refine ⟨(preguard M R h p.batch p.row s).val,?_,?_⟩
    · change run (machine M R) (code h) (entry h p.batch p.row s)=_
      rw [complete_code,run_append]
      simp only [(preguard M R h p.batch p.row s).property,Option.bind_some]
      apply Signed.Control.single
      apply Signed.Control.skip
      simpa only [sentinel,ne_self_iff_false,decide_false] using condition_eval M R h p.batch p.row s
    · simp only [sentinel,if_true]
      rfl
  · have nonnegative := valid.resolve_left sentinel
    have metadata : (data s).indices p.batch=(((data s).indices p.batch).toNat:Int) :=
      (Int.toNat_of_nonneg nonnegative).symm
    have positiveN : 0<h.N := by have := p.rowInside;omega
    let slot := ((data s).indices p.batch).toNat
    refine ⟨(iteration M R h p.batch p.row slot s metadata).val,?_,?_⟩
    · change run (machine M R) (code h) (entry h p.batch p.row s)=_
      rw [complete_code,run_append]
      simp only [(preguard M R h p.batch p.row s).property,Option.bind_some]
      apply Signed.Control.single
      apply Signed.Control.pass
      · simpa [sentinel] using condition_eval M R h p.batch p.row s
      · rw [complete_active,run_append,(preloop M R h p.batch p.row slot s metadata).property,Option.bind_some]
        apply Signed.Control.single
        apply Signed.Control.loop_one
        · exact preloop_length M R h p.batch p.row slot s metadata positiveN
        · exact (iteration M R h p.batch p.row slot s metadata).property
    · simpa only [sentinel,if_false] using iteration_memory M R h p.batch p.row slot s metadata positiveN

#axiomsClean source_executes

#axiomsClean preguard
#axiomsClean preloop
#axiomsClean iteration
#axiomsClean iteration_memory
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
