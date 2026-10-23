import bench.tritonbench_g.matmul_tma.Proofs.Native

/-! Numerical ports for exactly one source `tl.dot` and its output conversion.
The dot port consumes one logical FP16 row and column and returns one FP32
value. It has no access to memory, allocation, launch or a final output buffer.
Its internal native accumulation order and rounding are deliberately not
identified with the separately rounded sequential software interpreter. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.Primitive
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

structure Numerics (inner : Nat) (α : Type) where
  dot : (Fin inner → α) → (Fin inner → α) → α
  castHalf : α → α
  castSingle : α → α

/-- Local primitive contracts only. The source supplies FP16 operands, FP32
accumulation, no explicit incoming accumulator, and empty dot options.
These laws do not assume a wrapper result or a software/native graph rewrite. -/
structure Laws {α : Type} (h : Shape) (M : Algebra α) (D : Dot α)
    (Q : Numerics h.inner α) : Prop where
  dot : ∀ (a : TileIndex [h.rows,h.inner] → α)
      (b : TileIndex [h.inner,h.columns] → α) (i : TileIndex [h.rows,h.columns]),
    D.apply (some .fp32) {} (batch := []) a b i =
      Q.dot (fun k => a (i.1,k,PUnit.unit)) (fun k => b (k,i.2.1,PUnit.unit))
  halfCast : ∀ x, M.cast none .fp32 .fp16 x = Q.castHalf x
  singleCast : ∀ x, M.cast none .fp32 .fp32 x = Q.castSingle x

/-- Independent initial-memory expression: one row/column primitive, followed
by exactly the selected scalar output conversion. -/
def reference {α : Type} [Inhabited α] (h : Shape) (Q : Numerics h.inner α)
    (f : Flags) (r : Regions) (s : State α) (i : TileIndex [h.rows,h.columns]) : α :=
  let dot := Q.dot
    (fun k => aValues (parameters h f r) s (i.1,k,PUnit.unit))
    (fun k => bValues (parameters h f r) s (k,i.2.1,PUnit.unit))
  if f.half then Q.castHalf dot else Q.castSingle dot

theorem output_eq {α : Type} [Inhabited α] (h : Shape) (M : Algebra α) (D : Dot α)
    (Q : Numerics h.inner α) (law : Laws h M D Q) (f : Flags) (r : Regions)
    (s : State α) :
    outputValues M D (parameters h f r) f.half s = reference h Q f r s := by
  funext i
  unfold outputValues reference
  have hd := law.dot (aValues (parameters h f r) s) (bValues (parameters h f r) s) i
  change M.cast none .fp32 (outputDType f.half)
    (D.apply (some .fp32) {} (batch := []) (m := h.rows) (k := h.inner) (n := h.columns)
      (aValues (parameters h f r) s) (bValues (parameters h f r) s) i) = _
  rw [hd]
  cases hh : f.half <;> simp only [outputDType, hh, Bool.false_eq_true, ↓reduceIte]
  · exact law.singleCast _
  · exact law.halfCast _

#axiomsClean output_eq
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Primitive
