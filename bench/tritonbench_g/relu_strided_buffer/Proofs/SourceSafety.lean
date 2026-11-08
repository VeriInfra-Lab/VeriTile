import bench.tritonbench_g.relu_strided_buffer.Proofs.TileSafety

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.Access
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Body extracted from the actual grid-stride source. -/
def loopBody (c : Parameters) : List ComputeStmt :=
  PastedTile.iterations c

def atTile {α : Type} (s : State α) (P C j : Nat) : State α :=
  (s.setReg "j" .nat [] (fun _ => j)).setReg "tile_id" .nat [] (fun _ => P+j*C)

theorem loop_head_runs {α : Type} [Inhabited α] (M : Algebra α) (c : Parameters)
    (s : State α) (P C j : Nat) (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) :
    run M ((loopBody c).take 1) (s.setReg "j" .nat [] (fun _ => j))=
      some (atTile s P C j) := by
  simp [loopBody,PastedTile.iterations,PastedTile.grid,gridStrideKernel,originalKernel,ComputeKernel.surfaceBody,List.take,run,step,evalExpr,
    evalOp_unfold,numeric,hp,hc,atTile]
  rfl

theorem loop_body_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (c : Parameters) (s : State α) (P C j : Nat)
    (branch : c.oneTile = Bool.false) (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    Safety.traceSafeList M bounds (loopBody c) (s.setReg "j" .nat [] (fun _ => j)) := by
  rw [←List.take_append_drop 1 (loopBody c),Safety.traceSafeList_append]
  constructor
  · simp [loopBody,PastedTile.iterations,PastedTile.grid,gridStrideKernel,originalKernel,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric,hp,hc]
  · intro t ran
    have eq := Option.some.inj ((loop_head_runs M c s P C j hp hc).symm.trans ran)
    subst t
    have tail : (loopBody c).drop 1=perTile c := by
      simp [loopBody,perTile,PastedTile.body,branch]
    rw [tail]
    exact tile_safe M cmp compare c (P+j*C) (atTile s P C j) (by simp [atTile]) bounds input output

theorem loop_body_preserves {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (c : Parameters) (s : State α) (P C j : Nat)
    (branch : c.oneTile = Bool.false) (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) (t : State α)
    (ran : run M (loopBody c) (s.setReg "j" .nat [] (fun _ => j))=some t) :
    t.regs .nat [] "pid"=some (fun _ => P) ∧ t.regs .nat [] "num_ctas"=some (fun _ => C) := by
  obtain ⟨u,hu,_,up,uc⟩ := tile_implementation M cmp compare c (P+j*C)
    (atTile s P C j) (by simp [atTile])
  rw [←List.take_append_drop 1 (loopBody c),run_append,loop_head_runs M c s P C j hp hc,
    Option.bind_some] at ran
  have tail : (loopBody c).drop 1=perTile c := by
    simp [loopBody,perTile,PastedTile.body,branch]
  rw [tail] at ran
  have eq := Option.some.inj (hu.symm.trans ran)
  subst t
  rw [up,uc]
  simp [atTile,hp,hc]

def selected (folded : Bool) (c : Parameters) : ComputeKernel :=
  if folded then foldedKernel c else kernel c

/-- Proof-only flattening of the immutable source branch. -/
def controlBody (c : Parameters) : List ComputeStmt :=
  (kernel c).surfaceBody.take 2 ++
    (if c.oneTile then PastedTile.oneTile c else PastedTile.grid c)

def prefixState {α : Type} (c : Parameters) (s : State α) : State α :=
  let t := s.setReg "pid" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "num_tiles0" .nat [] (fun _ => (c.n+c.B-1)/c.B)
  if c.oneTile then t.setReg "tile_id" .nat [] (fun _ => s.pids 0)
  else t.setReg "num_ctas" .nat [] (fun _ => s.numPids 0)

theorem prefix_runs {α : Type} [Inhabited α] (M : Algebra α) (folded : Bool)
    (c : Parameters) (s : State α) :
    run M ((controlBody c).take 3) s=some (prefixState c s) := by
  cases folded <;> cases branch : c.oneTile <;>
    simp [controlBody,PastedTile.oneTile,PastedTile.grid,kernel,foldedKernel,branch,originalKernel,FP.Structural.bop,oneTileKernel,oneTileFolded,
      gridStrideKernel,gridStrideFolded,ComputeKernel.surfaceBody,List.take,run,step,
      evalExpr,evalOp_unfold,numeric,prefixState] <;> rfl

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α) (folded : Bool)
    (c : Parameters) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds ((controlBody c).take 3) s := by
  cases folded <;> cases branch : c.oneTile <;>
    simp [controlBody,PastedTile.oneTile,PastedTile.grid,kernel,foldedKernel,branch,originalKernel,FP.Structural.bop,oneTileKernel,oneTileFolded,
      gridStrideKernel,gridStrideFolded,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric]

/-- Both original branches and the existing integer-folded candidate obey
their signed input/output footprints throughout the complete actual loop. -/
theorem control_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (c : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    Safety.traceSafeList M bounds (controlBody c) s := by
  rw [←List.take_append_drop 3 (controlBody c),Safety.traceSafeList_append]
  refine ⟨prefix_safe M folded c s bounds,?_⟩
  intro t ran
  have eq := Option.some.inj ((prefix_runs M folded c s).symm.trans ran)
  subst t
  cases branch : c.oneTile
  · have tail : (controlBody c).drop 3=
        [.forRange "j" 0 c.tiles 1 (loopBody c)] := by
      simp [controlBody,PastedTile.grid,kernel,branch,gridStrideKernel,originalKernel,
        gridStrideFolded,ComputeKernel.surfaceBody,loopBody,PastedTile.iterations]
    rw [tail,Safety.traceSafeList_cons]
    refine ⟨?_,fun _ _ => by simp only [Safety.traceSafeList_unfold]⟩
    simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,true_and]
    apply Safety.rangeSafe_of_invariant M bounds "j" c.tiles 1 (loopBody c)
      (fun _ t => t.regs .nat [] "pid"=some (fun _ => s.pids 0) ∧
        t.regs .nat [] "num_ctas"=some (fun _ => s.numPids 0))
    · rintro j _ t ⟨hp,hc⟩
      exact loop_body_safe M cmp compare c t (s.pids 0) (s.numPids 0) j branch hp hc bounds input output
    · rintro j _ t ⟨hp,hc⟩ u hu
      exact loop_body_preserves M cmp compare c t (s.pids 0) (s.numPids 0) j branch hp hc u hu
    · simp [prefixState,branch]
  · have tail : (controlBody c).drop 3=perTile c := by
      simp [controlBody,PastedTile.oneTile,PastedTile.body,perTile,kernel,branch,oneTileKernel,originalKernel,
        oneTileFolded,ComputeKernel.surfaceBody,perTile]
    rw [tail]
    exact tile_safe M cmp compare c (s.pids 0) (prefixState c s)
      (by simp [prefixState,branch]) bounds input output

/-- Flatten only the immutable constexpr branch; both raw source arms remain
in Kernels. This is a safety equivalence, not a change to the source authority. -/
private theorem constant_branch {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (flag : Bool) (pre yes no : List ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds (pre++[.ifThenElse (.constBool flag) yes no]) s ↔
      Safety.traceSafeList M bounds (pre++(if flag then yes else no)) s := by
  simp only [Safety.traceSafeList_append]
  cases flag <;> simp [Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeOp_unfold,evalOp_unfold]

private theorem source_layout (folded : Bool) (c : Parameters) :
    (selected folded c).surfaceBody = (selected folded c).surfaceBody.take 2 ++
      [.ifThenElse (.constBool c.oneTile) (PastedTile.oneTile c) (PastedTile.grid c)] := by
  cases folded <;> cases branch : c.oneTile <;>
    simp [selected,kernel,foldedKernel,branch,oneTileKernel,gridStrideKernel,
      oneTileFolded,gridStrideFolded,originalKernel,candidateKernel,ComputeKernel.surfaceBody,
      PastedTile.oneTile,PastedTile.grid]

private theorem source_control_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (folded : Bool) (c : Parameters) (s : State α) :
    Safety.traceSafeList M bounds (selected folded c).surfaceBody s ↔
      Safety.traceSafeList M bounds (controlBody c) s := by
  rw [source_layout,constant_branch]
  simp only [controlBody,Safety.traceSafeList_append]
  cases folded <;> cases branch : c.oneTile <;>
    simp [selected,kernel,foldedKernel,branch,oneTileKernel,gridStrideKernel,
      oneTileFolded,gridStrideFolded,originalKernel,candidateKernel,ComputeKernel.surfaceBody,
      List.take,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,run,step,evalExpr,evalOp_unfold,numeric,FP.Structural.bop] <;> rfl

/-- Actual source: both dispatch arms and their complete grid-stride loops,
including the candidate's host-folded integer assignment. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (c : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    TraceSafe M bounds (selected folded c) s :=
  (source_control_safe M bounds folded c s).mpr
    (control_safe M cmp compare folded c s bounds input output)

#axiomsClean control_safe
#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.ReluStridedBuffer.Access
