import bench.tritonbench_g.matrix_transpose.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Parameters where
  input : RegionName
  output : RegionName
  sx : Nat
  sy : Nat
  ox : Nat
  oy : Nat
  m : Nat
  d : Nat

/-- The existing full fp16 source, including the existing index candidate. -/
def source (v : Bool) (p : Parameters) : ComputeKernel :=
  typedKernel v p.input p.output p.sx p.sy p.ox p.oy p.m p.d

/-- Independent output-indexed transpose addresses. -/
def inputOffset (p : Parameters) (i : TileIndex [p.d, p.m]) : Nat :=
  i.2.1.val * p.sx + i.1.val * p.sy

def outputOffset (p : Parameters) (i : TileIndex [p.d, p.m]) : Nat :=
  i.1.val * p.ox + i.2.1.val * p.oy

/-- Every output bit word is read from initial memory, before any store. -/
def outputValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.d, p.m]) : α :=
  (s.mem p.input (inputOffset p i)).read .fp16

/-- Pure memory specification, with no interpreter or execution result. -/
def referenceMemory {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.d, p.m]).foldl (fun mem i => fun r o =>
    if r = p.output ∧ o = outputOffset p i then .mk .fp16 (outputValue p s i)
    else mem r o) s.mem

/-- Register-level witness of the actual six-statement execution. All tile
loads precede the output fold, even when the input and output regions alias. -/
def execution {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  let r0 := s.setReg "size_m_arange" .nat [p.m] (fun i => i.1.val)
  let r1 := r0.setReg "d_head_arange" .nat [p.d] (fun i => i.1.val)
  let r2 := r1.setReg "matrix_ptr" .ptr [p.m,p.d]
    (fun i => (p.input, i.1.val * p.sx + i.2.1.val * p.sy))
  let r3 := r2.setReg "out_ptr" .ptr [p.d,p.m]
    (fun i => (p.output, outputOffset p i))
  let r4 := r3.setReg "matrix" .fp16 [p.m,p.d]
    (fun i => (s.mem p.input (i.1.val * p.sx + i.2.1.val * p.sy)).read .fp16)
  (TileShape.allIndices [p.d,p.m]).foldl
    (fun t i => t.write p.output (outputOffset p i) (.mk .fp16 (outputValue p s i))) r4

theorem execute {α : Type} [Inhabited α] (M : Algebra α)
    (v : Bool) (p : Parameters) (s : State α) :
    exec M (source v p) s = some (execution p s) := by
  cases v <;>
    simp [source, typedKernel, ComputeKernel.surfaceBody, exec, run, step, evalExpr, evalOp, store,
      execution, inputOffset, outputOffset, outputValue, State.setReg, bop, numeric,
      transposeValues, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.insertAxis, TileShape.dropInsertedIndex, Region.cast, Nat.mul_comm]
  all_goals
    congr 1
    congr 1
    funext dtype shape name
    split_ifs <;> simp_all [bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.mul_comm]
  all_goals
    subst_vars
    funext i
    simp [bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.mul_comm]

theorem scatter_memory {α ι : Type} (xs : List ι) (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => t.write region (offset i) (value i)) s).mem =
      xs.foldl (fun mem i => fun r o =>
        if r = region ∧ o = offset i then value i else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons, ih]; rfl

theorem execution_memory {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    (execution p s).mem = referenceMemory p s := by
  simp [execution, scatter_memory, referenceMemory]

theorem readback {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (hOutInj : Function.Injective (outputOffset p)) (i : TileIndex [p.d,p.m]) :
    (execution p s).mem p.output (outputOffset p i) =
      .mk .fp16 (outputValue p s i) := by
  exact State.scatter_readback _ _ _ _ i hOutInj

theorem frame {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : TileIndex [p.d,p.m], o ≠ outputOffset p i) :
    (execution p s).mem r o = s.mem r o := by
  unfold execution
  exact State.scatter_frame _ _ _ _ r o
    (hmiss.imp_right (fun h i _ => h i)) _

/-- No floating primitive, cast, padding value or arbitrary backend callback
can affect this complete source execution. -/
theorem backend_independent {α : Type} [Inhabited α] (M N : Algebra α)
    (v : Bool) (p : Parameters) (s : State α) :
    exec M (source v p) s = exec N (source v p) s := by
  rw [execute, execute]

#axiomsClean execute
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
#axiomsClean backend_independent
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
