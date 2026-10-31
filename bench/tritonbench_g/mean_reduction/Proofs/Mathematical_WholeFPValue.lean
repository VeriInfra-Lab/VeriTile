import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_SourceDefinitions
import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_WholeFP
import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_WholeGrid

namespace VeriTile.Bench.TritonBenchG.MeanReduction.WholeFPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def codePrefix (X Mean : RegionName) (M N : Nat) : List ComputeStmt :=
  (source X Mean M N).surfaceBody.take 5

def body (N : Nat) : List ComputeStmt :=
  match ((source "X" "Mean" 0 N).surfaceBody[5]? : Option ComputeStmt) with
  | some (.forRange _ _ _ _ code) => code
  | _ => []

def suffix (N : Nat) : List ComputeStmt := (source "X" "Mean" 0 N).surfaceBody.drop 6

theorem complete_source (X Mean : RegionName) (M N : Nat) :
    (source X Mean M N).surfaceBody = codePrefix X Mean M N ++
      [.forRange "off" 0 N 8 (body N)] ++ suffix N := by
  rfl

def prefixState {α : Type} (A : Algebra α) (X Mean : RegionName) (M N : Nat)
    (pid : Nat) (s : State α) : State α :=
  let s := s.setReg "pid" .nat [8,1] (fun i => pid*8+i.1.val)
  let s := s.setReg "X" .ptr [8,1] (fun i => (X,(pid*8+i.1.val)*N))
  let s := s.setReg "Mean" .ptr [8,1] (fun i => (Mean,pid*8+i.1.val))
  let s := s.setReg "row_mask" .bool [8,1] (fun i => decide (pid*8+i.1.val<M))
  s.setReg "_mean" .real [8,8] (fun _ => A.literal (some .fp32) .real 0)

theorem prefix_runs {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (s : State α) :
    run A (codePrefix X Mean M N) s = some (prefixState A X Mean M N (s.pids 0) s) := by
  simp [codePrefix,source,typedKernel, VeriTile.Bench.Repairs.MeanReduction.sourceKernel, VeriTile.Bench.Repairs.MeanReduction.indexCommutedSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,
    Region.cast,prefixState,Option.bind]
  congr 10 <;> (first | rfl | (funext i; simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]))

abbrev Acc (α : Type) := Values α .real [8,8]

def Context {α : Type} (X Mean : RegionName) (M N pid : Nat) (memory : RegionName → Nat → Cell α)
    (acc : Acc α) (s : State α) : Prop :=
  s.regs .ptr [8,1] "X" = some (fun i => (X,(pid*8+i.1.val)*N)) ∧
  s.regs .ptr [8,1] "Mean" = some (fun i => (Mean,pid*8+i.1.val)) ∧
  s.regs .bool [8,1] "row_mask" = some (fun i => decide (pid*8+i.1.val<M)) ∧
  s.regs .real [8,8] "_mean" = some acc ∧ s.mem = memory

def chunk {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName) (M N pid off : Nat)
    (memory : RegionName → Nat → Cell α) : Acc α :=
  fun i => A.cast (some .fp32) .real .real
    (if pid*8+i.1.val<M ∧ off+i.2.1.val<N then
      (memory X ((pid*8+i.1.val)*N+(off+i.2.1.val))).read .real
     else A.literal (some .fp32) .real 0)

def nextAcc {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α) : Acc α :=
  fun i => A.binary (some .fp32) .real .add (acc i) (chunk A X M N pid off memory i)

def bodyState {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α) (s : State α) : State α :=
  let s := s.setReg "off" .nat [] (fun _ => off)
  let s := s.setReg "cols" .nat [1,8] (fun i => off+i.2.1.val)
  let s := s.setReg "col_mask" .bool [1,8] (fun i => decide (off+i.2.1.val<N))
  let s := s.setReg "mask" .bool [8,8]
    (fun i => decide (pid*8+i.1.val<M) && decide (off+i.2.1.val<N))
  let s := s.setReg "a" .real [8,8] (chunk A X M N pid off memory)
  s.setReg "_mean" .real [8,8] (nextAcc A X M N pid off memory acc)

theorem body_runs {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α) (s : State α)
    (ctx : Context X Mean M N pid memory acc s) :
    run A (body N) (s.setReg "off" .nat [] (fun _ => off)) =
      some (bodyState A X M N pid off memory acc s) := by
  rcases ctx with ⟨hX,hMean,hRow,hAcc,hMem⟩
  simp [body,source,typedKernel, VeriTile.Bench.Repairs.MeanReduction.sourceKernel, VeriTile.Bench.Repairs.MeanReduction.indexCommutedSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,
    Region.cast,bodyState,chunk,nextAcc,Option.bind,hX,hMean,hRow,hAcc,hMem]
  simp [ComputeDType.eraseDType,ofFloat,toFloat,hAcc,Option.bind,chunk,nextAcc]
  congr 10 <;> (first | rfl | (funext ⟨i,j,k⟩; cases k; norm_num [nextAcc,chunk,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]))

theorem body_context {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α) (s : State α)
    (ctx : Context X Mean M N pid memory acc s) :
    Context X Mean M N pid memory (nextAcc A X M N pid off memory acc)
      (bodyState A X M N pid off memory acc s) := by
  rcases ctx with ⟨hX,hMean,hRow,hAcc,hMem⟩
  simp [Context,bodyState,hX,hMean,hRow,hMem,State.setReg]

def accumulated {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (off : Nat) (acc : Acc α) : Acc α :=
  if h : off < N then
    accumulated A X M N pid memory (off+8) (nextAcc A X M N pid off memory acc)
  else acc
termination_by N-off

/-- The source range terminates with exactly the ordered fp32 accumulator recurrence. -/
theorem range_runs {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (off : Nat) (acc : Acc α)
    (s : State α) (ctx : Context X Mean M N pid memory acc s) :
    ∃ t, range A "off" off N 8 (body N) s = some t ∧
      Context X Mean M N pid memory (accumulated A X M N pid memory off acc) t := by
  by_cases h : off<N
  · have cb := body_context A X Mean M N pid off memory acc s ctx
    obtain ⟨t,ht,hc⟩ := range_runs A X Mean M N pid memory (off+8)
      (nextAcc A X M N pid off memory acc) _ cb
    refine ⟨t,?_,?_⟩
    · rw [range]
      simp only [show ¬(8:Nat)=0 by omega,if_false,h,if_true,
        body_runs A X Mean M N pid off memory acc s ctx,Option.bind_some]
      exact ht
    · rw [accumulated]; simpa only [h,if_pos] using hc
  · refine ⟨s,?_,?_⟩
    · rw [range]; simp [h]
    · rw [accumulated]; simpa only [h,if_neg] using ctx
termination_by N-off

theorem prefix_context {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (s : State α) :
    Context X Mean M N (s.pids 0) s.mem (fun _ => A.literal (some .fp32) .real 0)
      (prefixState A X Mean M N (s.pids 0) s) := by
  simp [Context,prefixState]

/-- The backend sum is the precise source primitive, followed by its fp32 division. -/
def meanValues {α : Type} (A : Algebra α) (N : Nat) (acc : Acc α) : Values α .real [8] :=
  fun i => A.binary (some .fp32) .real .div
    (A.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false acc i)
    (A.literal (some .fp32) .real (N:ℝ))

def suffixState {α : Type} (A : Algebra α) (Mean : RegionName) (M N pid : Nat)
    (acc : Acc α) (s : State α) : State α :=
  let s := s.setReg "mean" .real [8] (meanValues A N acc)
  let s := s.setReg "mean" .real [8,1] (fun i => meanValues A N acc (i.1,PUnit.unit))
  (TileShape.allIndices [8,1]).foldl (fun t i =>
    if pid*8+i.1.val<M then t.write Mean (pid*8+i.1.val)
      (.mk .real (meanValues A N acc (i.1,PUnit.unit))) else t) s

theorem suffix_runs {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α) (s : State α)
    (ctx : Context X Mean M N pid memory acc s) :
    run A (suffix N) s = some (suffixState A Mean M N pid acc s) := by
  rcases ctx with ⟨hX,hMean,hRow,hAcc,hMem⟩
  simp [suffix,source,typedKernel, VeriTile.Bench.Repairs.MeanReduction.sourceKernel, VeriTile.Bench.Repairs.MeanReduction.indexCommutedSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,store,
    Region.cast,suffixState,meanValues,Option.bind,hMean,hRow,hAcc]
  simp [ComputeDType.eraseDType,Option.bind,hMean,hRow,meanValues]
  rfl

theorem row_injective (pid : Nat) :
    Function.Injective (fun i : TileIndex [8,1] => pid*8+i.1.val) := by
  rintro ⟨a,b,u⟩ ⟨c,d,v⟩ he
  have hac : a=c := Fin.ext (by simpa using he)
  subst c
  have hbd : b=d := Subsingleton.elim _ _
  subst d
  cases u; cases v; rfl

theorem suffix_value {α : Type} (A : Algebra α) (Mean : RegionName) (M N pid : Nat)
    (acc : Acc α) (s : State α) (i : Fin 8) (hi : pid*8+i.val<M) :
    (suffixState A Mean M N pid acc s).mem Mean (pid*8+i.val) =
      .mk .real (meanValues A N acc (i,PUnit.unit)) := by
  apply State.masked_scatter_readback Mean (fun i : TileIndex [8,1] => pid*8+i.1.val)
    (fun i => Cell.mk .real (meanValues A N acc (i.1,PUnit.unit)))
    (fun i => pid*8+i.1.val<M) _ (i,⟨0,by omega⟩,PUnit.unit) hi (row_injective pid)

theorem suffix_frame {α : Type} (A : Algebra α) (Mean : RegionName) (M N pid : Nat)
    (acc : Acc α) (s : State α) (r : RegionName) (o : Nat)
    (h : r≠Mean ∨ o<pid*8 ∨ pid*8+8≤o ∨ M≤o) :
    (suffixState A Mean M N pid acc s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · right
    intro i _ hi
    have := i.1.isLt
    rcases h with h | h | h <;> omega

/-- Explicit primitive formula, independent of program execution and output memory. -/
def tileValue {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName) (M N pid : Nat)
    (memory : RegionName → Nat → Cell α) (lane : Fin 8) : α :=
  meanValues A N (accumulated A X M N pid memory 0
    (fun _ => A.literal (some .fp32) .real 0)) (lane,PUnit.unit)

theorem accumulated_congr {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid : Nat) (memory other : RegionName → Nat → Cell α)
    (same : ∀ i, (memory X i).read .real = (other X i).read .real)
    (off : Nat) (acc : Acc α) :
    accumulated A X M N pid memory off acc = accumulated A X M N pid other off acc := by
  conv_lhs => rw [accumulated]
  conv_rhs => rw [accumulated]
  split
  · have stepEq : nextAcc A X M N pid off memory acc = nextAcc A X M N pid off other acc := by
      funext i
      simp only [nextAcc,chunk,same]
    rw [stepEq]
    exact accumulated_congr A X M N pid memory other same (off+8) _
  · rfl
termination_by N-off

theorem tileValue_congr {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid : Nat) (memory other : RegionName → Nat → Cell α)
    (same : ∀ i, (memory X i).read .real = (other X i).read .real) (lane : Fin 8) :
    tileValue A X M N pid memory lane = tileValue A X M N pid other lane := by
  unfold tileValue
  rw [accumulated_congr A X M N pid memory other same]

theorem tile_result {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (s : State α) :
    ∃ final, exec A (source X Mean M N) s = some final ∧
      (∀ i : Fin 8, s.pids 0*8+i.val<M → final.mem Mean (s.pids 0*8+i.val) =
        .mk .real (tileValue A X M N (s.pids 0) s.mem i)) ∧
      (∀ r o, (r≠Mean ∨ ∀ i : Fin 8, s.pids 0*8+i.val<M → o≠s.pids 0*8+i.val) →
        final.mem r o = s.mem r o) := by
  obtain ⟨middle,loopRan,ctx⟩ := range_runs A X Mean M N (s.pids 0) s.mem 0
    (fun _ => A.literal (some .fp32) .real 0) _ (prefix_context A X Mean M N s)
  let acc := accumulated A X M N (s.pids 0) s.mem 0 (fun _ => A.literal (some .fp32) .real 0)
  refine ⟨suffixState A Mean M N (s.pids 0) acc middle,?_,?_,?_⟩
  · change run A (source X Mean M N).surfaceBody s = _
    rw [complete_source,run_append,run_append,prefix_runs]
    simp only [Option.bind_some,List.singleton_append,run,step,loopRan]
    exact suffix_runs A X Mean M N (s.pids 0) s.mem acc middle ctx
  · intro i hi
    exact suffix_value A Mean M N (s.pids 0) acc middle i hi
  · intro r o outside
    have hm : middle.mem = s.mem := ctx.2.2.2.2
    rw [← hm]
    unfold suffixState
    apply State.masked_scatter_frame
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun i _ hi => ho i.1 hi

def rowValue {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName) (M N : Nat)
    (memory : RegionName → Nat → Cell α) (row : Nat) : α :=
  tileValue A X M N (row/8) memory ⟨row%8,by omega⟩

theorem rowValue_at {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (i : Fin 8) :
    rowValue A X M N memory (pid*8+i.val) = tileValue A X M N pid memory i := by
  have hp : (pid*8+i.val)/8=pid := by omega
  have hi : (⟨(pid*8+i.val)%8,by omega⟩ : Fin 8)=i := Fin.ext (by change (pid*8+i.val)%8=i.val; omega)
  simp only [rowValue,hp,hi]

theorem tiles_result {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (separate : X≠Mean) (tiles : List Nat) (s : State α) :
    ∃ final, WholeFP.runTiles A Bool.false X Mean M N tiles s = some final ∧
      (∀ i, i<M → Whole.Covered tiles i → final.mem Mean i =
        .mk .real (rowValue A X M N s.mem i)) ∧
      (∀ r o, (r≠Mean ∨ ∀ pid∈tiles, ∀ i : Fin 8,
        pid*8+i.val<M → o≠pid*8+i.val) → final.mem r o = s.mem r o) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · rintro i _ ⟨pid,hpid,_⟩; simp at hpid
    · intro r o _; rfl
  | cons pid rest ih =>
    obtain ⟨middle,ran,values,frame⟩ := tile_result A X Mean M N (WholeFP.tileState pid s)
    obtain ⟨final,tail,valuesTail,frameTail⟩ := ih middle
    have inputFrame (i : Nat) : (middle.mem X i).read .real = (s.mem X i).read .real := by
      rw [frame X i (Or.inl separate)]
      rfl
    refine ⟨final,?_,?_,?_⟩
    · change (exec A (source X Mean M N) (WholeFP.tileState pid s)).bind _ = _
      rw [ran]
      exact tail
    · intro i hi covered
      by_cases hc : Whole.Covered rest i
      · rw [valuesTail i hi hc]
        congr 1
        exact tileValue_congr A X M N (i/8) middle.mem s.mem inputFrame _
      · have preserve : final.mem Mean i = middle.mem Mean i := by
          apply frameTail Mean i
          right
          intro q hq lane active equal
          exact hc ⟨q,hq,lane,equal⟩
        rw [preserve]
        rcases covered with ⟨q,hq,lane,equal⟩
        rcases List.mem_cons.mp hq with head | tail
        · subst q; subst i
          rw [rowValue_at]
          exact values lane hi
        · exact False.elim (hc ⟨q,tail,lane,equal⟩)
    · intro r o outside
      rw [frameTail r o (by
        rcases outside with hr | ho
        · exact Or.inl hr
        · exact Or.inr fun q hq lane active => ho q (List.mem_cons_of_mem pid hq) lane active)]
      apply frame r o
      rcases outside with hr | ho
      · exact Or.inl hr
      · exact Or.inr fun lane active => ho pid List.mem_cons_self lane active

/-- All output cells follow the explicit primitive formula for any complete
schedule, including zero programs; the entire exterior memory is unchanged. -/
theorem grid_result {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (separate : X≠Mean) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (Whole.gridSize M))) (s : State α) :
    ∃ final, WholeFP.runTiles A Bool.false X Mean M N tiles s = some final ∧
      (∀ i, i<M → final.mem Mean i = .mk .real (rowValue A X M N s.mem i)) ∧
      (∀ r o, (r≠Mean ∨ M≤o) → final.mem r o = s.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := tiles_result A X Mean M N separate tiles s
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    apply values i hi
    obtain ⟨pid,hpid,lane,equal⟩ := Whole.grid_covers M i hi
    exact ⟨pid,schedule.mem_iff.mpr hpid,lane,equal⟩
  · intro r o outside
    apply frame r o
    rcases outside with hr | ho
    · exact Or.inl hr
    · right; intro pid hpid lane active; omega

theorem accumulated_congr_active {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N pid : Nat) (memory other : RegionName → Nat → Cell α)
    (same : ∀ i, i<M*N → (memory X i).read .real = (other X i).read .real)
    (off : Nat) (acc : Acc α) :
    accumulated A X M N pid memory off acc = accumulated A X M N pid other off acc := by
  conv_lhs => rw [accumulated]
  conv_rhs => rw [accumulated]
  split
  · have stepEq : nextAcc A X M N pid off memory acc = nextAcc A X M N pid off other acc := by
      funext i
      unfold nextAcc chunk
      split
      · rename_i h
        rw [same _ (by nlinarith [h.1,h.2])]
      · rfl
    rw [stepEq]
    exact accumulated_congr_active A X M N pid memory other same (off+8) _
  · rfl
termination_by N-off

theorem rowValue_congr_active {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M N : Nat) (memory other : RegionName → Nat → Cell α)
    (same : ∀ i, i<M*N → (memory X i).read .real = (other X i).read .real) (row : Nat) :
    rowValue A X M N memory row = rowValue A X M N other row := by
  unfold rowValue tileValue
  rw [accumulated_congr_active A X M N (row/8) memory other same]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.WholeFPValue
