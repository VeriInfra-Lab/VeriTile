import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.TileSafety

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.Access
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Body extracted from the actual grid-stride source. -/
def loopBody (c : Parameters) : List ComputeStmt :=
  match ((gridStrideKernel c).surfaceBody[2]? : Option ComputeStmt) with
  | some (.ifThenElse _ _ no) =>
      match (no[1]? : Option ComputeStmt) with
      | some (.forRange _ _ _ _ body) => body
      | _ => []
  | _ => []

def atTile {α : Type} (s : State α) (P C j : Nat) : State α :=
  (s.setReg "j" .nat [] (fun _ => j)).setReg "tile_id" .nat [] (fun _ => P+j*C)

theorem loop_head_runs {α : Type} [Inhabited α] (M : Algebra α) (c : Parameters)
    (s : State α) (P C j : Nat) (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) :
    run M ((loopBody c).take 1) (s.setReg "j" .nat [] (fun _ => j))=
      some (atTile s P C j) := by
  simp [loopBody,gridStrideKernel,originalBody,foldedBody,ComputeKernel.surfaceBody,List.take,run,step,evalExpr,
    evalOp_unfold,numeric,hp,hc,atTile]
  rfl

theorem loop_body_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (c : Parameters) (s : State α) (P C j : Nat)
    (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    Safety.traceSafeList M bounds (loopBody c) (s.setReg "j" .nat [] (fun _ => j)) := by
  rw [←List.take_append_drop 1 (loopBody c),Safety.traceSafeList_append]
  constructor
  · simp [loopBody,gridStrideKernel,originalBody,foldedBody,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric,hp,hc]
  · intro t ran
    have eq := Option.some.inj ((loop_head_runs M c s P C j hp hc).symm.trans ran)
    subst t
    exact tile_safe M cmp compare c (P+j*C) (atTile s P C j) (by simp [atTile]) bounds input output

theorem loop_body_preserves {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (c : Parameters) (s : State α) (P C j : Nat)
    (hp : s.regs .nat [] "pid"=some (fun _ => P))
    (hc : s.regs .nat [] "num_ctas"=some (fun _ => C)) (t : State α)
    (ran : run M (loopBody c) (s.setReg "j" .nat [] (fun _ => j))=some t) :
    t.regs .nat [] "pid"=some (fun _ => P) ∧ t.regs .nat [] "num_ctas"=some (fun _ => C) := by
  obtain ⟨u,hu,_,up,uc⟩ := tile_implementation M cmp compare c (P+j*C)
    (atTile s P C j) (by simp [atTile])
  rw [←List.take_append_drop 1 (loopBody c),run_append,loop_head_runs M c s P C j hp hc,
    Option.bind_some] at ran
  have eq := Option.some.inj (hu.symm.trans ran)
  subst t
  rw [up,uc]
  simp [atTile,hp,hc]

def selected (folded : Bool) (c : Parameters) : ComputeKernel :=
  if folded then foldedKernel c else kernel c

def sourceBranches (folded : Bool) (c : Parameters) : List ComputeStmt × List ComputeStmt :=
  match ((selected folded c).surfaceBody[2]? : Option ComputeStmt) with
  | some (.ifThenElse _ yes no) => (yes,no)
  | _ => ([],[])

/-- A proof decomposition of the chosen constexpr branch, extracted from
the actual writer. It is never installed as an authoritative kernel or IO. -/
def chosenBody (folded : Bool) (c : Parameters) : List ComputeStmt :=
  (selected folded c).surfaceBody.take 2 ++
    (if c.oneTile then (sourceBranches folded c).1 else (sourceBranches folded c).2)

private theorem actual_body (folded : Bool) (c : Parameters) :
    (selected folded c).surfaceBody =
      (selected folded c).surfaceBody.take 2 ++
        [.ifThenElse (.constBool c.oneTile) (sourceBranches folded c).1 (sourceBranches folded c).2] := by
  cases folded <;> rfl

/-- A constant branch checks precisely its selected body. The unused arm has
no execution or memory-safety obligation. -/
private theorem branch_trace {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (headBody yes no : List ComputeStmt) (b : Bool) (s : State α) :
    Safety.traceSafeList M bounds (headBody ++ [.ifThenElse (.constBool b) yes no]) s ↔
      Safety.traceSafeList M bounds (headBody ++ (if b then yes else no)) s := by
  have single : ∀ t, Safety.traceSafeList M bounds [.ifThenElse (.constBool b) yes no] t ↔
      Safety.traceSafeList M bounds (if b then yes else no) t := by
    intro t
    rw [Safety.traceSafeList_cons]
    cases b <;> simp [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold]
  simp only [Safety.traceSafeList_append,single]


def prefixState {α : Type} (c : Parameters) (s : State α) : State α :=
  let t := s.setReg "pid" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "num_tiles0" .nat [] (fun _ => (c.n+c.B-1)/c.B)
  if c.oneTile then t.setReg "tile_id" .nat [] (fun _ => s.pids 0)
  else t.setReg "num_ctas" .nat [] (fun _ => s.numPids 0)

theorem prefix_runs {α : Type} [Inhabited α] (M : Algebra α) (folded : Bool)
    (c : Parameters) (s : State α) :
    run M ((chosenBody folded c).take 3) s=some (prefixState c s) := by
  cases folded <;> cases branch : c.oneTile <;>
    simp [chosenBody,sourceBranches,selected,kernel,foldedKernel,branch,oneTileKernel,oneTileFolded,
      gridStrideKernel,gridStrideFolded,originalBody,foldedBody,ComputeKernel.surfaceBody,List.take,run,step,
      evalExpr,evalOp_unfold,numeric,prefixState] <;> rfl

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α) (folded : Bool)
    (c : Parameters) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds ((chosenBody folded c).take 3) s := by
  cases folded <;> cases branch : c.oneTile <;>
    simp [chosenBody,sourceBranches,selected,kernel,foldedKernel,branch,oneTileKernel,oneTileFolded,
      gridStrideKernel,gridStrideFolded,originalBody,foldedBody,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric]

/-- Both original branches and the existing integer-folded candidate obey
their signed input/output footprints throughout the complete actual loop. -/
private theorem chosen_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (c : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    Safety.traceSafeList M bounds (chosenBody folded c) s := by
  rw [←List.take_append_drop 3 (chosenBody folded c),Safety.traceSafeList_append]
  refine ⟨prefix_safe M folded c s bounds,?_⟩
  intro t ran
  have eq := Option.some.inj ((prefix_runs M folded c s).symm.trans ran)
  subst t
  cases branch : c.oneTile
  · have tail : (chosenBody folded c).drop 3=
        [.forRange "j" 0 c.tiles 1 (loopBody c)] := by
      cases folded <;> simp [chosenBody,sourceBranches,selected,kernel,foldedKernel,branch,gridStrideKernel,
        gridStrideFolded,originalBody,foldedBody,ComputeKernel.surfaceBody,loopBody]
    rw [tail,Safety.traceSafeList_cons]
    refine ⟨?_,fun _ _ => by simp only [Safety.traceSafeList_unfold]⟩
    simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,true_and]
    apply Safety.rangeSafe_of_invariant M bounds "j" c.tiles 1 (loopBody c)
      (fun _ t => t.regs .nat [] "pid"=some (fun _ => s.pids 0) ∧
        t.regs .nat [] "num_ctas"=some (fun _ => s.numPids 0))
    · rintro j _ t ⟨hp,hc⟩
      exact loop_body_safe M cmp compare c t (s.pids 0) (s.numPids 0) j hp hc bounds input output
    · rintro j _ t ⟨hp,hc⟩ u hu
      exact loop_body_preserves M cmp compare c t (s.pids 0) (s.numPids 0) j hp hc u hu
    · simp [prefixState,branch]
  · have tail : (chosenBody folded c).drop 3=perTile c := by
      cases folded <;> simp [chosenBody,sourceBranches,selected,kernel,foldedKernel,branch,oneTileKernel,
        oneTileFolded,originalBody,foldedBody,ComputeKernel.surfaceBody,perTile]
    rw [tail]
    exact tile_safe M cmp compare c (s.pids 0) (prefixState c s)
      (by simp [prefixState,branch]) bounds input output

/-- Safety of the complete actual original/candidate writer, including the
constexpr condition and both selected execution shapes. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (c : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    TraceSafe M bounds (selected folded c) s := by
  have safe := chosen_safe M cmp compare folded c s bounds input output
  change Safety.traceSafeList M bounds (selected folded c).surfaceBody s
  rw [actual_body folded c]
  exact (branch_trace M bounds _ _ _ c.oneTile s).mpr safe

#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.ReluStridedBuffer.Access
