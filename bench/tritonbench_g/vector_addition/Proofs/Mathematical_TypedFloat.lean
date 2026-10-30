import bench.tritonbench_g.vector_addition.Proofs.Mathematical_HostGrid

/-! Typed materialization of the original elementwise load/add/store graph.
The five storage formats below have distinct existing AST channels. Operand
promotion and the conversion back to x's output dtype are explicit. This is
not fp64 or fnuz coverage, nor proof of native compiler/allocator support.
Conversions between distinct formats and floating addition are opaque
numerical primitives; identity conversions are defined as identity.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters offset)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | fp16 | bf16 | fp32 | f8e4 | f8e5
  deriving DecidableEq, Repr

def Storage.dtype : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32
  | .f8e4 => .f8e4 | .f8e5 => .f8e5

/-- Triton 3.7.1 tensor/tensor floating promotion, restricted to the five
represented storage formats. The original output is empty_like(x). -/
def promote (x y : Storage) : Storage :=
  if x = .fp32 ∨ y = .fp32 then .fp32
  else if x = .fp16 ∨ y = .fp16 then .fp16
  else if x = .bf16 ∧ y = .bf16 then .bf16
  else if x = .bf16 ∨ y = .bf16 then .fp32
  else if x = y then x else .fp16

def castBackend {α : Type} (M : Algebra α) : Algebra α :=
  { M with cast := fun p src dst a => if src = dst then a else M.cast p src dst a }

def castValue {α : Type} (M : Algebra α) (src dst : FloatDType) (a : α) : α :=
  if src = dst then a else M.cast none src dst a

/-- All eight original statements, including both loads before the store.
Explicit casts materialize implicit tensor promotion and pointer-store cast.
The .real channel is allowed by this generic constructor but is not used by
any Storage profile and must not be interpreted as fp64 provenance. -/
def materialized (dx dy dc : FloatDType) (p : Parameters) : ComputeKernel :=
  .mk [p.x, p.y] [p.output] [
    .assign .nat [] "pid" (.alg (.programId 0)),
    .assign .nat [] "block_start" (.alg (.mul .nat .nil
      (.ref .nat [] "pid") (.constNat p.block))),
    .assign .nat [p.block] "offsets" (.alg (.add .nat .scalarL
      (.ref .nat [] "block_start") (.arange p.block))),
    .assign .bool [p.block] "mask" (.alg (.lt .nat .scalarR
      (.ref .nat [p.block] "offsets") (.constNat p.n))),
    .assign dx.toTileDType [p.block] "x" (.alg (.load dx.toTileDType
      (.region p.x (.ref .nat [p.block] "offsets"))
      (.mask (.ref .bool [p.block] "mask")) .zero)),
    .assign dy.toTileDType [p.block] "y" (.alg (.load dy.toTileDType
      (.region p.y (.ref .nat [p.block] "offsets"))
      (.mask (.ref .bool [p.block] "mask")) .zero)),
    .assign dc.toTileDType [p.block] "output" (.alg
      (.add dc.numericDType (.consSame .nil)
        (.castFloat dx dc (.ref dx.toTileDType [p.block] "x"))
        (.castFloat dy dc (.ref dy.toTileDType [p.block] "y")))),
    .store dx.toTileDType [p.block]
      (.region p.output (.ref .nat [p.block] "offsets"))
      (.alg (.castFloat dc dx (.ref dc.toTileDType [p.block] "output")))
      (.mask (.ref .bool [p.block] "mask")) ]

def source (x y : Storage) (p : Parameters) :=
  materialized x.dtype y.dtype (promote x y).dtype p

@[simp] theorem toFloat_ofFloat {α : Type} (d : FloatDType) (a : α) :
    toFloat d (ofFloat d a) = a := by cases d <;> rfl

@[simp] theorem ofFloat_toFloat {α : Type} (d : FloatDType) (a : Value α d.toTileDType) :
    ofFloat d (toFloat d a) = a := by cases d <;> rfl

@[simp] theorem toFloat_numeric {α : Type} (M : Algebra α) (d : FloatDType)
    (a b : Value α d.toTileDType) :
    toFloat d (numeric M none .add d.numericDType a b) =
      M.binary none d .add (toFloat d a) (toFloat d b) := by
  cases d <;> rfl

@[simp] theorem numeric_nat_add {α : Type} (M : Algebra α) (a b : Nat) :
    numeric M none .add NumericDType.nat a b = a+b := rfl

@[simp] theorem numeric_nat_mul {α : Type} (M : Algebra α) (a b : Nat) :
    numeric M none .mul NumericDType.nat a b = a*b := rfl

def resultCell {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk dx.toTileDType (ofFloat dx (castValue M dc dx
    (M.binary none dc .add
      (castValue M dx dc (toFloat dx ((mem p.x o).read dx.toTileDType)))
      (castValue M dy dc (toFloat dy ((mem p.y o).read dy.toTileDType))))))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then
        resultCell M dx dy dc p s.mem (offset p s i) else mem r o
    else mem) s.mem

/-- Actual complete structural execution; success is constructed, not assumed. -/
def execution {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (castBackend M) (materialized dx dy dc p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [materialized, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numericLt, natLt, bop, store, Region.cast, castBackend]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    (execution M dx dy dc p s).val.mem = expectedMemory M dx dy dc p s := by
  simp [execution, expectedMemory, offset, resultCell, bop,
    Broadcast.leftIndex, Broadcast.rightIndex,
    ValueProofs.masked_scatter_memory, State.setReg, castValue, castBackend]
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * p.block + i.1.val < p.n
  · simp only [hi, ↓reduceIte]
  · simp only [hi, ↓reduceIte]

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    expectedMemory M dx dy dc p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell M dx dy dc p s.mem (offset p s i)) else t) s).mem := by
  rw [ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory M dx dy dc p s p.output (offset p s i) = resultCell M dx dy dc p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M dx dy dc p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (ValueProofs.offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : ValueProofs.Outside p s r o) :
    expectedMemory M dx dy dc p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M dx dy dc p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell M dx dy dc p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M dx dy dc p s i (by rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      unfold offset at hi ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (castBackend M) (materialized dx dy dc p) s = some t ∧ t.mem = expectedMemory M dx dy dc p s :=
  ⟨(execution M dx dy dc p s).val, (execution M dx dy dc p s).property, execution_memory M dx dy dc p s⟩

/-- Erasing dtype and implicit conversions preserves the original body. This
checks addresses, masks and statement/register order, not native lowering. -/
theorem source_projection (x y : Storage) (p : Parameters) :
    (source x y p).eraseDType.toAlgKernel.body = (ValueProofs.source p).toAlgKernel.body := by
  cases x <;> cases y
  all_goals simp only [source, promote, Storage.dtype, materialized, ValueProofs.source,
    add_kernel, ComputeKernel.eraseDType, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, FloatDType.numericDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType, ComputeDType.eraseDType,
    Region.cast, Except.bind, bind]
  all_goals rfl

#specNonCircular expectedMemory avoiding [source, materialized, FP.Structural.exec]
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean source_projection
#axiomsClean expected_cell
#axiomsClean complete_correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
