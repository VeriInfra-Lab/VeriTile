import bench.tritonbench_g.mean_reduction.Proofs.SourceFormula
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPValue
import bench.tritonbench_g.mean_reduction.Proofs.WholeFinite

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

variable {α : Type} [Inhabited α] {rank : Nat} {x : Tensor rank} {dims : List (Fin rank)}
  {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
  {c : Contiguous α x dims nodup entry preexisting}

def Allocated.fpState (a : Allocated c) : State α where
  mem := fun r i => Cell.mk .real (if r="X" then a.memory (c.base+i) else a.memory (a.base+i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax=0 then gridSize (rows x.shape dims) else 1
  undef := fun dtype _ _ => defaultValue dtype

@[simp] theorem fpState_input (a : Allocated c) (i : Nat) :
    (a.fpState.mem "X" i).read .real = a.memory (c.base+i) := by simp [Allocated.fpState]

@[simp] theorem fpState_output (a : Allocated c) (i : Nat) :
    (a.fpState.mem "Mean" i).read .real = a.memory (a.base+i) := by simp [Allocated.fpState]

theorem allocated_flattened (a : Allocated c) (positive : 0<columns x.shape dims)
    (i : Nat) (hi : i<rows x.shape dims*columns x.shape dims) :
    a.memory (c.base+i) = flattenedInput x dims nodup entry positive i := by
  have hc := a.input_copy
    ⟨i/columns x.shape dims,(Nat.div_lt_iff_lt_mul positive).mpr hi⟩
    ⟨i%columns x.shape dims,Nat.mod_lt _ positive⟩
  have index : c.base+(i/columns x.shape dims)*columns x.shape dims+i%columns x.shape dims =
      c.base+i := by nlinarith [Nat.mod_add_div i (columns x.shape dims)]
  simpa only [flattenedInput,dif_pos hi,index] using hc

theorem allocated_fp_result (A : Algebra α) (a : Allocated c)
    (positive : 0<columns x.shape dims) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize (rows x.shape dims)))) :
    ∃ final, WholeFP.runTiles A Bool.false "X" "Mean" (rows x.shape dims)
      (columns x.shape dims) tiles a.fpState = some final ∧
      (∀ row : Fin (rows x.shape dims),
        a.commit (fun i => (final.mem "Mean" i).read .real) (a.base+row.val) =
          fpFormula A x dims nodup entry positive row.val) ∧
      (∀ addr, (∀ i, i<rows x.shape dims → addr≠a.base+i) →
        a.commit (fun i => (final.mem "Mean" i).read .real) addr = a.memory addr) ∧
      (∀ addr, preexisting addr →
        a.commit (fun i => (final.mem "Mean" i).read .real) addr = entry addr) := by
  obtain ⟨final,ran,values,frame⟩ := WholeFPValue.grid_result A "X" "Mean"
    (rows x.shape dims) (columns x.shape dims) (by decide) tiles schedule a.fpState
  have exterior (i : Nat) (hi : rows x.shape dims ≤ i) :
      (final.mem "Mean" i).read .real = a.memory (a.base+i) := by
    rw [frame "Mean" i (Or.inr hi),fpState_output]
  have physicalFrame := a.commit_frame (fun i => (final.mem "Mean" i).read .real) exterior
  refine ⟨final,ran,?_,physicalFrame,?_⟩
  · intro row
    rw [a.commit_value,values row.val row.isLt,Cell.read_same]
    apply WholeFPValue.rowValue_congr_active
    intro i hi
    simpa only [fpState_input,Cell.read_same] using allocated_flattened a positive i hi
  · intro addr hp
    rw [physicalFrame addr (by
      intro i hi equal
      apply a.fresh i hi
      simpa only [← equal] using hp)]
    exact a.protected_frame addr hp

def executeFP (A : Algebra α) (variant : Bool)
    (rt : Runtime α x dims nodup entry preexisting) (keepdim : Bool) : Option (Output α) := do
  let c ← rt.compress
  let a ← rt.allocate c
  if rt.launchReady c a then
    let final ← WholeFP.runTiles A variant "X" "Mean" (rows x.shape dims)
      (columns x.shape dims) (rt.order c a) a.fpState
    return makeOutput a keepdim (fun i => (final.mem "Mean" i).read .real)
  else none

def FPHostPost (A : Algebra α) (entry : Nat → α) (x : Tensor rank) (dims : List (Fin rank))
    (nodup : dims.Nodup) (positive : 0<columns x.shape dims) (keepdim : Bool)
    (preexisting : Nat → Prop) (out : Output α) : Prop :=
  out.shape=outputShape x.shape dims keepdim ∧ out.dtype=.fp32 ∧ out.device=x.device ∧
  (∀ row : Fin (rows x.shape dims), out.memory (out.base+row.val) =
    fpFormula A x dims nodup entry positive row.val) ∧
  (∀ addr, preexisting addr → out.memory addr=entry addr)

theorem executeFP_returns (A : Algebra α) (rt : Runtime α x dims nodup entry preexisting)
    (keepdim : Bool) (positive : 0<columns x.shape dims)
    (c : Contiguous α x dims nodup entry preexisting) (a : Allocated c)
    (hc : rt.compress=some c) (ha : rt.allocate c=some a)
    (ready : rt.launchReady c a=Bool.true) :
    ∃ out, executeFP A Bool.false rt keepdim=some out ∧
      FPHostPost A entry x dims nodup positive keepdim preexisting out := by
  obtain ⟨final,ran,values,_frame,preexistingFrame⟩ :=
    allocated_fp_result A a positive (rt.order c a) (rt.schedule c a)
  refine ⟨makeOutput a keepdim (fun i => (final.mem "Mean" i).read .real),
    ?_,rfl,rfl,rfl,values,preexistingFrame⟩
  simp [executeFP,hc,ha,ready,ran]

/-- Original and address-commuted wrappers have identical failures and complete
returned state for every numerical algebra and shared ordinary runtime. -/
theorem executeFP_equal (A : Algebra α) (rt : Runtime α x dims nodup entry preexisting)
    (keepdim : Bool) : executeFP A Bool.false rt keepdim=executeFP A Bool.true rt keepdim := by
  unfold executeFP
  simp only [WholeFP.grid_equal]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
