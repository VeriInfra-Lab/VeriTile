import bench.tritonbench_g.mean_reduction.Proofs.WholeLowered
import bench.tritonbench_g.mean_reduction.Proofs.WholePublic

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

variable {rank : Nat} {x : Tensor rank} {dims : List (Fin rank)} {nodup : dims.Nodup}
  {entry : Nat → BitVec 32} {preexisting : Nat → Prop}
  {c : Contiguous (BitVec 32) x dims nodup entry preexisting}

/-- Original tensor coordinates feed a fully explicit lowered graph. Each
output may select its own actual shared-store winning lane. No tree equality. -/
def nativeFormula (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → BitVec 32) (positive : 0<columns x.shape dims) (row : Nat) : BitVec 32 :=
  WholeLowered.tileFormula B "X" (rows x.shape dims) (columns x.shape dims) (row/8)
    (fun _ i => Cell.mk .real (flattenedInput x dims nodup entry positive i))
    (fun lane => winner ((row/8)*8+lane.val)) ⟨row%8,by omega⟩

theorem fp_native_formula (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (positive : 0<columns x.shape dims) (row : Nat) :
    fpFormula (WholeLowered.family B (columns x.shape dims) winner (row/8))
      x dims nodup entry positive row=nativeFormula B winner x dims nodup entry positive row := by
  unfold fpFormula WholeFPValue.rowValue WholeLowered.family nativeFormula
  exact WholeLowered.tileFormula_eq B (columns x.shape dims) positive _ "X" _ _ _ _

theorem allocated_native_result (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (a : Allocated c) (positive : 0<columns x.shape dims) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize (rows x.shape dims)))) :
    ∃ final, WholeFPFamily.runTiles (WholeLowered.family B (columns x.shape dims) winner)
      Bool.false "X" "Mean" (rows x.shape dims) (columns x.shape dims) tiles a.fpState=some final ∧
      (∀ row : Fin (rows x.shape dims),
        a.commit (fun i => (final.mem "Mean" i).read .real) (a.base+row.val)=
          nativeFormula B winner x dims nodup entry positive row.val) ∧
      (∀ addr, (∀ i, i<rows x.shape dims → addr≠a.base+i) →
        a.commit (fun i => (final.mem "Mean" i).read .real) addr=a.memory addr) ∧
      (∀ addr, preexisting addr →
        a.commit (fun i => (final.mem "Mean" i).read .real) addr=entry addr) := by
  obtain ⟨final,ran,values,frame⟩ := WholeFPFamily.grid_result
    (WholeLowered.family B (columns x.shape dims) winner) "X" "Mean"
    (rows x.shape dims) (columns x.shape dims) (by decide) tiles schedule a.fpState
  have exterior (i : Nat) (hi : rows x.shape dims ≤ i) :
      (final.mem "Mean" i).read .real=a.memory (a.base+i) := by
    rw [frame "Mean" i (Or.inr hi),fpState_output]
  have physicalFrame := a.commit_frame (fun i => (final.mem "Mean" i).read .real) exterior
  refine ⟨final,ran,?_,physicalFrame,?_⟩
  · intro row
    rw [a.commit_value,values row.val row.isLt,Cell.read_same,← fp_native_formula B winner positive row.val]
    apply WholeFPValue.rowValue_congr_active
    intro i hi
    simpa only [fpState_input,Cell.read_same] using allocated_flattened a positive i hi
  · intro addr hp
    rw [physicalFrame addr (by
      intro i hi equal
      apply a.fresh i hi
      simpa only [← equal] using hp)]
    exact a.protected_frame addr hp

def executeNative (B : WholeLowered.Scalars) (winner : Nat → Fin 8) (variant : Bool)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool) : Option (Output (BitVec 32)) := do
  let c ← rt.compress
  let a ← rt.allocate c
  if rt.launchReady c a then
    let final ← WholeFPFamily.runTiles (WholeLowered.family B (columns x.shape dims) winner)
      variant "X" "Mean" (rows x.shape dims) (columns x.shape dims) (rt.order c a) a.fpState
    return makeOutput a keepdim (fun i => (final.mem "Mean" i).read .real)
  else none

theorem native_outcomes (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool)
    (positive : 0<columns x.shape dims) :
    HostOutcomes (fun row => nativeFormula B winner x dims nodup entry positive row.val)
      rt keepdim (executeNative B winner Bool.false rt keepdim) := by
  cases hc : rt.compress with
  | none =>
    simp only [bind,Option.bind,executeNative,hc,Option.bind_none,HostOutcomes]
    exact Or.inl hc
  | some c =>
    cases ha : rt.allocate c with
    | none =>
      simp only [bind,Option.bind,executeNative,hc,ha,Option.bind_some,Option.bind_none,HostOutcomes]
      exact Or.inr ⟨c,hc,Or.inl ha⟩
    | some a =>
      cases ready : rt.launchReady c a with
      | «false» =>
        simp only [bind,Option.bind,executeNative,hc,ha,ready,Option.bind_some,Bool.false_eq_true,if_false,HostOutcomes]
        exact Or.inr ⟨c,hc,Or.inr ⟨a,ha,ready⟩⟩
      | «true» =>
        obtain ⟨final,ran,values,frame,protectedFrame⟩ :=
          allocated_native_result B winner a positive (rt.order c a) (rt.schedule c a)
        simp only [bind,Option.bind,executeNative,hc,ha,ready,Option.bind_some,if_true,ran,HostOutcomes]
        exact ⟨c,a,hc,ha,ready,rfl,rfl,rfl,rfl,values,frame,protectedFrame⟩

theorem native_equal (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool) :
    executeNative B winner Bool.false rt keepdim=executeNative B winner Bool.true rt keepdim := by
  unfold executeNative
  simp only [WholeFPFamily.grid_equal]

/-- Native acceptance candidate: all source/host/finite obligations are proved;
only actual per-node scalar add, signed32 conversion and div.full require
numerical binding. N=1 contains add(load,+0), its distinct explicit tree, and
no division. Different replica/profile trees need not agree. -/
specification mean_dim_native_fp32_source_outcomes (B : WholeLowered.Scalars)
    (winner : Nat → Fin 8) (_arguments : SourceArguments rank dims)
    (positive : 0<columns x.shape dims)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool)
    (finite : ∀ c a, rt.compress=some c → rt.allocate c=some a →
      rt.launchReady c a=Bool.true → FiniteAdmission a) :
    HostOutcomes (fun row => nativeFormula B winner x dims nodup entry positive row.val)
      rt keepdim (executeNative B winner Bool.false rt keepdim) ∧
    executeNative B winner Bool.false rt keepdim=executeNative B winner Bool.true rt keepdim ∧
    ∀ c a hc ha ready, IndexCorrespondence a (finite c a hc ha ready) := by
  exact ⟨native_outcomes B winner rt keepdim positive,native_equal B winner rt keepdim,
    fun c a hc ha ready => index_correspondence a (finite c a hc ha ready)⟩

#axiomsClean mean_dim_native_fp32_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
