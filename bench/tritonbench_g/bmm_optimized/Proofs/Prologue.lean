import bench.tritonbench_g.bmm_optimized.Proofs.Reference

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem setReg_numPids (s : RegisterState α) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Values α dtype shape) :
    (s.setReg name dtype shape v).numPids=s.numPids := rfl

@[simp] theorem bop_function {A B C : Type} {a b out : TileShape}
    (f : A → B → C) (bc : Broadcast a b out) (va : TileIndex a → A) (vb : TileIndex b → B) :
    FP.Structural.bop f bc va vb=(fun i=>f (va (bc.leftIndex i)) (vb (bc.rightIndex i))) := rfl

private abbrev RG := VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.groupSize

def gridResult (p : Parameters) (s : RegisterState α) : RegisterState α :=
  let t := s.setReg "pid_b" .nat [] (fun _=>s.pids 2)
  let t := t.setReg "A" .ptr [] (fun _=>(p.A,s.pids 2*p.M*p.K))
  let t := t.setReg "B" .ptr [] (fun _=>(p.B,s.pids 2*p.K*p.N))
  let t := t.setReg "O" .ptr [] (fun _=>(p.O,s.pids 2*p.M*p.N))
  let t := t.setReg "pidx" .nat [] (fun _=>s.pids 0)
  let t := t.setReg "pidy" .nat [] (fun _=>s.pids 1)
  if p.GROUP=1 then
    (t.setReg "pid_m" .nat [] (fun _=>s.pids 0)).setReg "pid_n" .nat [] (fun _=>s.pids 1)
  else
    let pid := s.pids 0+s.pids 1*s.numPids 0
    let count := s.numPids 1*p.GROUP
    let group := pid/count
    let t := t.setReg "gridx" .nat [] (fun _=>s.numPids 0)
    let t := t.setReg "gridy" .nat [] (fun _=>s.numPids 1)
    let t := t.setReg "pid" .nat [] (fun _=>pid)
    let t := t.setReg "num_CTA_per_group" .nat [] (fun _=>count)
    let t := t.setReg "group_id" .nat [] (fun _=>group)
    let t := t.setReg "inner_group_id" .nat [] (fun _=>pid%count)
    let t := t.setReg "GROUP_SIZE" .nat [] (fun _=>RG (s.numPids 0) group p.GROUP)
    (t.setReg "pid_m" .nat [] (fun _=>rowBlock p s)).setReg "pid_n" .nat [] (fun _=>columnBlock p s)

def tailResult (o : BmmOptimized.Scalar α) (p : Parameters) (s t : RegisterState α) : RegisterState α :=
  let t := t.setReg "offs_m" .nat [p.TM] (fun i=>row p s i.1)
  let t := t.setReg "offs_n" .nat [p.TN] (fun i=>column p s i.1)
  let t := t.setReg "offs_k" .nat [p.TK] (fun i=>i.1.val)
  let t := if !p.divM then t.setReg "mask_m" .bool [p.TM] (fun i=>decide (row p s i.1<p.M)) else t
  let t := if !p.divN then t.setReg "mask_n" .bool [p.TN] (fun i=>decide (column p s i.1<p.N)) else t
  let t := t.setReg "a_ptrs" .ptr [p.TM,p.TK] (fun i=>(p.A,aOffset p s 0 i))
  let t := t.setReg "b_ptrs" .ptr [p.TK,p.TN] (fun i=>(p.B,bOffset p s 0 i))
  let t := t.setReg "o_ptrs" .ptr [p.TM,p.TN] (fun i=>(p.O,oOffset p s i))
  let t := t.setReg "num_iters" .nat [] (fun _=>iterations p)
  t.setReg "o" .real [p.TM,p.TN] (fun _=>o.zero)

def preludeResult (o : BmmOptimized.Scalar α) (p : Parameters) (s : RegisterState α) :=
  tailResult o p s (gridResult p s)

def gridCode (p : Parameters) :=
  (prelude {p with divM:=Bool.false,divN:=Bool.false,divK:=Bool.false}).take 7

def tailCode (p : Parameters) := (prelude p).drop 7

theorem prelude_split (p : Parameters) : prelude p=gridCode p++tailCode p := by
  have eq : gridCode p=(prelude p).take 7 := by
    rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
    cases dm <;> cases dn <;> cases dk <;> rfl
  rw [eq,tailCode,List.take_append_drop]

private theorem gridExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (s : RegisterState α) :
    run o rest (gridCode p) s=some (gridResult p s) := by
  simp only [gridCode,prelude,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
    Option.map,Option.getD,beq_self_eq_true,↓reduceIte,List.take]
  by_cases hg : p.GROUP=1
  all_goals
    by_cases hb : (s.pids 0+s.pids 1*s.numPids 0)/(s.numPids 1*p.GROUP)*p.GROUP+p.GROUP>s.numPids 0
    all_goals
      simp [gridResult,RG,rowBlock,columnBlock,
        VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile,
        VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile,
        VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.groupSize,
        hg,hb,run,step,localStep,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
        numeric,FP.Structural.bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,Region.cast,
        Option.bind,integralFloorDiv,integralMod,algebra,Broadcast.leftIndex,Broadcast.rightIndex]
      try rfl

structure GridInvariant (p : Parameters) (s t : RegisterState α) : Prop where
  a : t.regs .ptr [] "A"=some (fun _=>(p.A,s.pids 2*p.M*p.K))
  b : t.regs .ptr [] "B"=some (fun _=>(p.B,s.pids 2*p.K*p.N))
  out : t.regs .ptr [] "O"=some (fun _=>(p.O,s.pids 2*p.M*p.N))
  m : t.regs .nat [] "pid_m"=some (fun _=>rowBlock p s)
  n : t.regs .nat [] "pid_n"=some (fun _=>columnBlock p s)

private theorem gridInvariant (p : Parameters) (s : RegisterState α) :
    GridInvariant p s (gridResult p s) := by
  by_cases hg : p.GROUP=1 <;> constructor <;>
    simp [gridResult,hg,rowBlock,columnBlock,
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile,
      VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile]

private theorem tail_k_flag (p : Parameters) :
    tailCode p=tailCode {p with divK:=Bool.false} := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> rfl

private theorem tailExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (s t : RegisterState α) (h : GridInvariant p s t) :
    run o rest (tailCode p) t=some (tailResult o p s t) := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  rw [tail_k_flag]
  change run o rest (tailCode ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,Bool.false⟩) t =
    some (tailResult o ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,Bool.false⟩ s t)
  cases dm <;> cases dn
  all_goals
    simp only [tailCode,prelude,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      Option.map,Option.getD,beq_self_eq_true,↓reduceIte,List.drop]
    simp [tailResult,row,column,aOffset,bOffset,oOffset,iterations,
      run,step,localStep,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,store,numeric,FP.Structural.bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
      Option.bind,integralFloorDiv,integralMod,algebra,Broadcast.leftIndex,Broadcast.rightIndex,
      h.a,h.b,h.out,h.m,h.n]
    try rfl

/-- The full source prefix succeeds from arbitrary initial registers. -/
theorem preludeExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (s : RegisterState α) :
    run o rest (prelude p) s=some (preludeResult o p s) := by
  rw [prelude_split,run_append,gridExecution,Option.bind_some]
  exact tailExecution o rest p s _ (gridInvariant p s)

#axiomsClean preludeExecution
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
