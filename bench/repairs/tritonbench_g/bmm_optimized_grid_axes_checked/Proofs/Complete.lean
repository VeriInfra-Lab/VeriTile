import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Body

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loopExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α) (k : Nat)
    (hk : k ≤ iterations p) (h : Invariant o p initial k current) :
    ∃ final,range o rest "_" k (iterations p) 1 (body p) current=some final ∧
      Invariant o p initial (iterations p) final := by
  have aux : ∀ fuel k (current : RegisterState α),iterations p-k=fuel → k ≤ iterations p →
      Invariant o p initial k current →
      ∃ final,range o rest "_" k (iterations p) 1 (body p) current=some final ∧
        Invariant o p initial (iterations p) final := by
    intro fuel
    induction fuel with
    | zero =>
      intro k current eq bound h
      have eq : k=iterations p := by omega
      subst k
      exact ⟨current,by rw [range,if_neg (by decide),if_neg (Nat.lt_irrefl _)],h⟩
    | succ fuel ih =>
      intro k current eq bound h
      have hk : k < iterations p := by omega
      have hset:=loopRegister o p initial current k h
      let next:=bodyResult o p initial (current.setReg "_" .nat [] (fun _=>k)) k
      have hn : Invariant o p initial (k+1) next := bodyInvariant o p initial _ k hset
      obtain ⟨final,run,hf⟩:=ih (k+1) next (by omega) (by omega) hn
      refine ⟨final,?_,hf⟩
      rw [range,if_neg (by decide),if_pos hk,bodyExecution o rest p initial _ k hset]
      exact run
  exact aux _ k current rfl hk h

def storeMaskResult (p : Parameters) (initial current : RegisterState α) : RegisterState α :=
  match p.divM,p.divN with
  | Bool.false,Bool.false => current.setReg "mask_c" .bool [p.TM,p.TN]
      (fun i=>decide (row p initial i.1<p.M ∧ column p initial i.2.1<p.N))
  | Bool.false,Bool.true => current.setReg "mask_c" .bool [p.TM,1]
      (fun i=>decide (row p initial i.1<p.M))
  | Bool.true,Bool.false => current.setReg "mask_c" .bool [1,p.TN]
      (fun i=>decide (column p initial i.2.1<p.N))
  | Bool.true,Bool.true => current

def scatter (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) : RegisterState α :=
  (TileShape.allIndices [p.TM,p.TN]).foldl (fun state i=>
    if active p initial i then state.write p.O (oOffset p initial i)
      (.mk .real (accumulation o p initial (iterations p) i)) else state) current

def epilogueResult (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) := scatter o p initial (storeMaskResult p initial current)

theorem epilogueExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α)
    (h : Invariant o p initial (iterations p) current) :
    run o rest (epilogue p) current=some (epilogueResult o p initial current) := by
  have hm:=h.maskM
  have hn:=h.maskN
  cases dm : p.divM <;> cases dn : p.divN <;> cases dk : p.divK
  all_goals
    simp only [dm,dn,Bool.false_eq_true,false_implies,true_implies] at hm hn
    simp [epilogue,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      dm,dn,dk,epilogueResult,scatter,storeMaskResult,active,
      run,step,localStep,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,store,numeric,FP.Structural.bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
      Option.bind,integralFloorDiv,integralMod,algebra,Broadcast.leftIndex,Broadcast.rightIndex,
      h.optrs,h.acc,hm,hn]
    try rfl

private theorem scatter_memory (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) (h : current.mem=initial.mem) :
    (scatter o p initial current).mem=(scatter o p initial initial).mem := by
  unfold scatter
  have aux : ∀ (lanes : List (TileIndex [p.TM,p.TN])) (left right : RegisterState α),
      left.mem=right.mem →
      (lanes.foldl (fun state i=>if active p initial i then state.write p.O (oOffset p initial i)
        (.mk .real (accumulation o p initial (iterations p) i)) else state) left).mem=
      (lanes.foldl (fun state i=>if active p initial i then state.write p.O (oOffset p initial i)
        (.mk .real (accumulation o p initial (iterations p) i)) else state) right).mem := by
    intro lanes
    induction lanes with
    | nil=>intro left right h;exact h
    | cons i lanes ih=>
      intro left right h
      simp only [List.foldl_cons]
      split_ifs <;> apply ih
      · simp [State.write,h]
      · exact h
  exact aux _ current initial h

theorem correctness (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (initial : RegisterState α) :
    ∃ final,exec o rest p initial=some final ∧
      final.mem=(scatter o p initial initial).mem := by
  let start:=preludeResult o p initial
  obtain ⟨middle,hm,hi⟩:=loopExecution o rest p initial start 0 (by omega)
    (preludeInvariant o p initial)
  let finish:=epilogueResult o p initial middle
  refine ⟨finish,?_,?_⟩
  · unfold exec
    rw [source_decomposition]
    have hp:=preludeExecution o rest p initial
    have hl : run o rest [.forRangeDyn "_" (.constNat 0) (.ref .nat [] "num_iters")
        (.constNat 1) (body p)] start=some middle := by
      have hstart:=preludeInvariant o p initial
      simpa [run,step,evalOp_unfold,hstart.iterations] using hm
    rw [run_append,run_append,hp,Option.bind_some,hl,Option.bind_some]
    exact epilogueExecution o rest p initial middle hi
  · apply scatter_memory
    cases dm : p.divM <;> cases dn : p.divN <;>
      simp [storeMaskResult,dm,dn,hi.memory]

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
