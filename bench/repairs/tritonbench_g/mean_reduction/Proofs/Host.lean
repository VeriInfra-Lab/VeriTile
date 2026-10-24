import bench.repairs.tritonbench_g.mean_reduction.Proofs.Contract
import bench.tritonbench_g.mean_reduction.Proofs.WholeNativeHost

namespace VeriTile.Bench.Repairs.MeanReduction
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary backing-storage bounds for the original input, compressed copy
and fresh output. Inactive masked pointers impose no padding-space condition. -/
structure StorageBounds {α : Type} {x : Tensor rank} {dims : List (Fin rank)}
    {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
    {c : Contiguous α x dims nodup entry preexisting} (a : Allocated c) : Prop where
  inputEnd : 4*(c.base+rows x.shape dims*columns x.shape dims) ≤ 2^64
  outputEnd : 4*(a.base+rows x.shape dims) ≤ 2^64
  original : ∀ i, 4*x.address i+4 ≤ 2^64 ∧ preexisting (x.address i)

theorem active_pointers {α : Type} {x : Tensor rank} {dims : List (Fin rank)}
    {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
    {c : Contiguous α x dims nodup entry preexisting} (a : Allocated c)
    (h : InputContract metadata x.shape dims dtype) (storage : StorageBounds a)
    (pid t lane col : Nat) (hp : pid < Whole.gridSize (rows x.shape dims))
    (ht : t < Whole.gridSize (columns x.shape dims)) (hl : lane < 8) (hc : col < 8)
    (row : pid*8+lane < rows x.shape dims) (column : t*8+col < columns x.shape dims) :
    (nativeInputPointer c.base pid lane (columns x.shape dims) (t*8) col).toNat =
      4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
    (nativeOutputPointer a.base pid lane).toNat=4*(a.base+pid*8+lane) := by
  exact checked_pointers h c.base a.base storage.inputEnd storage.outputEnd
    pid t lane col hp ht hl hc row column

def runReal {x : Tensor rank} {dims : List (Fin rank)} {nodup : dims.Nodup}
    {entry : Nat → ℝ} {preexisting : Nat → Prop}
    (metadata : TensorMetadata) (axes : Axes) (dtype : Option DType)
    (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool) :
    Except InputError (Option (Output ℝ)) := do
  let _ ← checkInput metadata x.shape axes dtype
  return executeReal rt keepdim

def runNative {x : Tensor rank} {dims : List (Fin rank)} {nodup : dims.Nodup}
    {entry : Nat → BitVec 32} {preexisting : Nat → Prop}
    (metadata : TensorMetadata) (axes : Axes) (dtype : Option DType)
    (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool) :
    Except InputError (Option (Output (BitVec 32))) := do
  let _ ← checkInput metadata x.shape axes dtype
  return executeNative B winner Bool.false rt keepdim

theorem real_outcomes {x : Tensor rank} {dims : List (Fin rank)} {nodup : dims.Nodup}
    {entry : Nat → ℝ} {preexisting : Nat → Prop}
    (h : Admitted metadata x.shape axes dtype dims)
    (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool) :
    ∃ result, runReal metadata axes dtype rt keepdim=.ok result ∧
      HostOutcomes (fun row=>mathematicalMean x.shape dims (x.read entry)
        ((batchLinear x.shape dims).symm row)) rt keepdim result := by
  refine ⟨executeReal rt keepdim,?_,Whole.real_outcomes rt keepdim⟩
  simp [runReal,(checkInput_iff metadata x.shape axes dtype dims).mpr h]

theorem native_outcomes {x : Tensor rank} {dims : List (Fin rank)} {nodup : dims.Nodup}
    {entry : Nat → BitVec 32} {preexisting : Nat → Prop}
    (h : Admitted metadata x.shape axes dtype dims) (B : WholeLowered.Scalars)
    (winner : Nat → Fin 8) (rt : Runtime (BitVec 32) x dims nodup entry preexisting)
    (keepdim : Bool) :
    ∃ result, runNative metadata axes dtype B winner rt keepdim=.ok result ∧
      HostOutcomes (fun row=>nativeFormula B winner x dims nodup entry h.2.positive row.val)
        rt keepdim result := by
  refine ⟨executeNative B winner Bool.false rt keepdim,?_,Whole.native_outcomes B winner rt keepdim h.2.positive⟩
  simp [runNative,(checkInput_iff metadata x.shape axes dtype dims).mpr h]

#axiomsClean active_pointers
#axiomsClean real_outcomes
#axiomsClean native_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction
