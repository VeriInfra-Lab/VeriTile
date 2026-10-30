import bench.tritonbench_g.int8_quantization.Proofs.SourceExecution
import bench.tritonbench_g.int8_quantization.Proofs.FP

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.FPTrace
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def constants (M : Algebra α) (p : Parameters) : Floating.Constants α :=
  ⟨scalar32 M 0.0,scalar32 M 1.0,scalar32 M (-1.0),scalar32 M 0.5,
    scalar32 M 127.0,scalar32 M (-127.0),
    M.literal none .real 340282346638528859811704183484516925440.0,scalar32 M p.factor⟩

/-- Extraction of exactly the production FP32 primitive calls. The upper
clamp uses the source minimum primitive, and the maximum preserves padded lane order. -/
def numerics (M : Algebra α) (ops : Model.Primitives α) : Floating.Numerics α where
  abs:=M.unary (some .fp32) (.abs .real)
  mul:=M.binary (some .fp32) .real .mul
  div:=M.binary (some .fp32) .real .div
  add:=M.binary (some .fp32) .real .add
  max:=M.binary (some .fp32) .real .max
  min:=ops.minimum
  le:=ops.le
  lt:=ops.less
  maximum:=fun {n} x=> M.reduceMax (some .fp32) (shape:=[n]) ⟨0,by simp⟩ Bool.false
    (fun i=> x i.1) PUnit.unit
  int8:=ops.int8

def inputs (M : Algebra α) (p : Parameters) (entry : State α) : Fin p.width→α :=
  fun j=> loaded M p entry (j,PUnit.unit)

/-- Two local context-normalization ports. Boolean expressions are emitted
with context none by this DSL; their reached FP32 operands and exact scalar
operations remain explicit. No execution or memory assumption occurs here. -/
structure Context (M : Algebra α) (p : Parameters) (entry : State α) : Prop where
  zero : M.literal none .real 0.0=scalar32 M 0.0
  abs : ∀ i,M.unary none (.abs .real) (transformed M p entry i)=
    M.unary (some .fp32) (.abs .real) (transformed M p entry i)

theorem bad_eq_zero (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    badValue M ops p positive entry=0 ↔ ∀ j : Fin p.width,flags M ops p entry (j,PUnit.unit)=Bool.true := by
  constructor
  · intro h j
    have bound:=Finset.le_sup' (s:=Finset.univ)
      (f:=fun j : Fin p.width=> if flags M ops p entry (j,PUnit.unit) then (0:Int) else 1)
      (Finset.mem_univ j)
    change (if flags M ops p entry (j,PUnit.unit) then (0:Int) else 1)≤ badValue M ops p positive entry at bound
    rw [h] at bound
    cases flag : flags M ops p entry (j,PUnit.unit) <;> simp [flag] at bound ⊢
  · intro h
    simp [badValue,h]

theorem bad_bool (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    decide (badValue M ops p positive entry=0)=
      !Floating.bad (fun j=> flags M ops p entry (j,PUnit.unit)) := by
  apply Bool.eq_iff_iff.mpr
  simp [bad_eq_zero,Floating.bad,List.any_eq_false]

theorem transformed_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) (i : Lane p) :
    transformed M p entry i=Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry) i.1 := by
  cases i with | mk j u=> cases u;rfl

theorem flags_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) (ctx : Context M p entry) :
    (fun j=> flags M ops p entry (j,PUnit.unit))=
      Floating.finite (constants M p) (numerics M ops)
        (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry)) := by
  funext j
  unfold flags
  rw [ctx.abs,transformed_trace M ops]
  rfl

theorem sanitized_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) (ctx : Context M p entry) (i : Lane p) :
    sanitized M ops p entry i=
      Floating.safe (constants M p)
        (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry))
        (Floating.finite (constants M p) (numerics M ops)
          (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry))) i.1 := by
  cases i with | mk j u=>
    cases u
    unfold sanitized
    rw [show flags M ops p entry (j,PUnit.unit)=_ from congrFun (flags_trace M ops p entry ctx) j,
      transformed_trace M ops]
    rfl

theorem scale_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) (ctx : Context M p entry) :
    scale M ops p entry=Floating.scale (constants M p) (numerics M ops)
      (Floating.safe (constants M p)
        (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry))
        (Floating.finite (constants M p) (numerics M ops)
          (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry)))) := by
  unfold scale maximum Floating.scale numerics
  simp only [sanitized_trace M ops p entry ctx]
  rfl

theorem usable_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) (ctx : Context M p entry) :
    usable M ops p positive entry=Floating.usable (constants M p) (numerics M ops)
      (Floating.finite (constants M p) (numerics M ops)
        (Floating.scaled (constants M p) (numerics M ops) p.query (inputs M p entry)))
      (scale M ops p entry) := by
  unfold usable
  rw [bad_bool,flags_trace M ops p entry ctx,ctx.zero]
  rfl

theorem arithmetic_trace (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) (ctx : Context M p entry) :
    ((fun j=> quantized M ops p positive entry (j,PUnit.unit)),savedScale M ops p positive entry)=
      Floating.run (constants M p) (numerics M ops) p.query (inputs M p entry) := by
  apply Prod.ext
  · funext j
    change quantized M ops p positive entry (j,PUnit.unit)=_
    unfold quantized biased clipped normalized quotient denominator
    simp only [usable_trace M ops p positive entry ctx,scale_trace M ops p entry ctx,
      sanitized_trace M ops p entry ctx]
    rfl
  · unfold savedScale
    have bad:=bad_bool M ops p positive entry
    rw [flags_trace M ops p entry ctx] at bad
    simp only [Floating.run]
    rw [←scale_trace M ops p entry ctx]
    by_cases hb : badValue M ops p positive entry=0
    all_goals
      simp [hb] at bad
      simp only [hb,bad,↓reduceIte,Bool.false_eq_true]
      all_goals rfl

/-- Complete source store readback equals the separately stated FP32 pipeline.
Every extra numerical hypothesis is one reached scalar/reduction primitive
from the frozen 14-port Local record, plus the two context ports above. -/
theorem source_fp (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) (ctx : Context M p entry)
    (reference : Floating.Numerics α)
    (laws : Floating.Local (constants M p) (numerics M ops) reference p.query (inputs M p entry))
    (i : Lane p) (active : Model.active p entry i) :
    (finalState M ops p positive entry).mem outRegion (Model.offset p entry i)=
      .mk .int ((Floating.run (constants M p) reference p.query (inputs M p entry)).1 i.1) ∧
    (finalState M ops p positive entry).mem scaleRegion (Model.group p entry)=
      .mk .real ((Floating.run (constants M p) reference p.query (inputs M p entry)).2) := by
  have trace:=(arithmetic_trace M ops p positive entry ctx).trans
    (Floating.local_equiv _ _ _ _ _ laws)
  constructor
  · rw [source_readback _ _ _ _ _ _ active]
    have value:=congrFun (congrArg Prod.fst trace) i.1
    cases i with | mk j u=> cases u;exact congrArg (Cell.mk .int) value
  · rw [source_scale]
    exact congrArg (Cell.mk .real) (congrArg Prod.snd trace)

#axiomsClean arithmetic_trace
#axiomsClean source_fp
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.FPTrace
