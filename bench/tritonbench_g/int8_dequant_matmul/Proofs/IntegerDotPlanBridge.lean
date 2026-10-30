import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteSemantics
import VeriTile.Triton.Float.IntegerDotPlan

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote.SharedInteger
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def width : WgmmaN → IntegerDot.WgmmaN
  | .n8 => .n8 | .n16 => .n16 | .n32 => .n32
  | .n64 => .n64 | .n128 => .n128 | .n256 => .n256
def oldWidth : IntegerDot.WgmmaN → WgmmaN
  | .n8 => .n8 | .n16 => .n16 | .n32 => .n32
  | .n64 => .n64 | .n128 => .n128 | .n256 => .n256
def isa : ISA → IntegerDot.ISA
  | .mmaM16N8K32Satfinite => .mmaM16N8K32Satfinite
  | .wgmmaM64K32 n => .wgmmaM64K32 (width n)
def oldISA : IntegerDot.ISA → ISA
  | .mmaM16N8K32Satfinite => .mmaM16N8K32Satfinite
  | .wgmmaM64K32 n => .wgmmaM64K32 (oldWidth n)

@[simp] theorem oldISA_isa (kind : ISA) : oldISA (isa kind) = kind := by
  cases kind with
  | mmaM16N8K32Satfinite => rfl
  | wgmmaM64K32 n => cases n <;> rfl

def micro : Micro k → IntegerDot.Micro k
  | .scalar lane => .scalar lane
  | .instruction kind lanes => .instruction (isa kind) lanes

@[simp] theorem micro_terms (op : Micro k) : (micro op).terms = op.terms := by
  cases op <;> rfl

def plan (p : DotPlan k) : IntegerDot.Plan k where
  operations := p.operations.map micro
  complete := by simpa only [List.flatMap_map,Function.comp_def,micro_terms] using p.complete
  fused := p.fused

def arithmetic (q : IntegerModel I F H) : IntegerDot.Arithmetic where
  normalize := fun z => q.decode (q.encode z)
  product8 := fun x y => q.decode (q.ops.product8 x y)
  add := fun x y => q.decode (q.ops.iadd (q.encode x) (q.encode y))
  mma32 := fun kind a b carry =>
    q.decode (q.ops.mma32 (oldISA kind) a b (q.encode carry))

theorem micro_eval (q : IntegerModel I F H) (op : Micro k)
    (left right : Fin k → BitVec 8) (carry : I) :
    (micro op).eval (arithmetic q).instructions left right (q.decode carry) =
      q.decode (op.eval q.ops left right carry) := by
  cases op <;>
    simp only [micro,IntegerDot.Micro.eval,IntegerDot.Arithmetic.instructions,
      arithmetic,Micro.eval,q.roundtrip,oldISA_isa]

theorem operations_eval (q : IntegerModel I F H) (operations : List (Micro k))
    (left right : Fin k → BitVec 8) (carry : I) :
    (operations.map micro).foldl (fun old op => op.eval (arithmetic q).instructions left right old)
        (q.decode carry) =
      q.decode (operations.foldl (fun old op => op.eval q.ops left right old) carry) := by
  induction operations generalizing carry with
  | nil => rfl
  | cons op rest ih => simp only [List.map_cons,List.foldl_cons,micro_eval,ih]

/-- Exact old representation boundary for every Int carry, including integers
outside the native accumulator range. No decode-encode identity is assumed. -/
theorem plan_eval (q : IntegerModel I F H) (p : DotPlan k)
    (left right : Fin k → BitVec 8) (old : Int) :
    (plan p).evalInt (arithmetic q) left right old =
      q.decode (p.eval q.ops left right (q.encode old)) := by
  unfold IntegerDot.Plan.evalInt DotPlan.eval
  simp only [plan]
  by_cases fused : p.fused = Bool.true
  · simp only [fused,ite_true]
    exact operations_eval q p.operations left right (q.encode old)
  · simp [fused]
    have folded := operations_eval q p.operations left right q.ops.izero
    rw [q.zero] at folded
    simp only [plan,arithmetic] at *
    rw [folded,q.roundtrip]

#axiomsClean plan_eval
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote.SharedInteger
