import bench.repairs.tritonbench_g.chunk_linear_attn.WholeReal
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeComposition
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

/-- Exact operation language used by the four selected all-FP32/IEEE kernels.
Casts and loads remain explicit materialization boundaries. -/
inductive Term where
  | bits : BitVec 32 → Term
  | literal : ℝ → Term
  | add : Term → Term → Term
  | mul : Term → Term → Term
  | cast : Term → Term
  | load : Term → Term
  | dot (k : Nat) : (Fin k → Term) → (Fin k → Term) → Term
instance : Inhabited Term := ⟨.literal 0⟩

def Term.meaning (decode : BitVec 32 → ℝ) : Term → ℝ
  | .bits b => decode b
  | .literal x => x
  | .add a b => a.meaning decode+b.meaning decode
  | .mul a b => a.meaning decode*b.meaning decode
  | .cast a | .load a => a.meaning decode
  | .dot k a b => ∑ e : Fin k, (a e).meaning decode*(b e).meaning decode

def algebra : Algebra Term where
  literal := fun _ _ x => .literal x
  negInf := .literal 0
  binary := fun _ _ op a b => match op with | .add => .add a b | .mul => .mul a b | _ => .literal 0
  unary := fun _ _ a => a
  cast := fun _ _ _ a => .cast a
  fromNat := fun _ n => .literal n
  fromInt := fun _ n => .literal n
  fp32Bits := fun b => .bits b.bits
  fp32Load := .load
  reduceMax := fun _ {_} _ _ _ _ => .literal 0
  reduceSum := fun _ {_} _ _ _ _ => .literal 0

def dotValues (l r out : FloatDType) : (batch : TileShape) → {m k n : Nat} →
    Values Term l.toTileDType (batch++[m,k]) → Values Term r.toTileDType (batch++[k,n]) →
    Values Term out.toTileDType (batch++[m,n])
  | [],_,k,_,a,b,i => ofFloat out (.dot k (fun e => toFloat l (a (i.1,e,()))) (fun e => toFloat r (b (e,i.2.1,()))))
  | _::ds,_,_,_,a,b,i => dotValues l r out ds (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
def dotPrimitive : DotPrimitive Term where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b

def realISA (decode : BitVec 32 → ℝ) : ISA ℝ where
  bits := decode
  literal := fun _ _ x => x
  binary := Real.algebra.binary
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fma := fun a b c => a*b+c
  mmaTF32 := fun _ _ a b c => c+∑ e : Fin 8, a e*b e

theorem fma_fold (xs : List (Fin k)) (a b : Fin k → ℝ) (seed : ℝ) :
    xs.foldl (fun acc e => a e*b e+acc) seed=seed+(xs.map (fun e => a e*b e)).sum := by
  induction xs generalizing seed with
  | nil => simp
  | cons e xs ih =>
    rw [List.foldl_cons,ih]
    simp only [List.map_cons,List.sum_cons]
    ring

theorem ieee_real (decode : BitVec 32 → ℝ) (k : Nat) (a b : Fin k → Graph) (seed : Graph) :
    (ieeeDot k a b seed).eval (realISA decode)=seed.eval (realISA decode)+
      ∑ e : Fin k, (a e).eval (realISA decode)*(b e).eval (realISA decode) := by
  rw [ieee_complete_chain]
  change (List.finRange k).foldl (fun acc e => (a e).eval (realISA decode)*(b e).eval (realISA decode)+acc)
    (seed.eval (realISA decode))=_
  rw [fma_fold]
  rw [← List.sum_toFinset _ (List.nodup_finRange _),List.toFinset_finRange]

def lower : Term → Graph
  | .bits b => .bits b
  | .literal x => .literal (some .fp32) .real x
  | .add a (.dot k x y) => ieeeDot k (fun e => lower (x e)) (fun e => lower (y e)) (lower a)
  | .add a (.mul b c) => .fma (lower b) (lower c) (lower a)
  | .add a b => .binary (some .fp32) .real .add (lower a) (lower b)
  | .mul a b => .binary (some .fp32) .real .mul (lower a) (lower b)
  | .cast a => .cast (some .fp32) .real .real (lower a)
  | .load a => .move .register (lower a)
  | .dot k a b => ieeeDot k (fun e => lower (a e)) (fun e => lower (b e)) VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero

theorem eval_zero (decode : BitVec 32 → ℝ) :
    Graph.eval (realISA decode) VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero=0 := rfl
 theorem eval_add (decode : BitVec 32 → ℝ) (a b : Graph) :
    Graph.eval (realISA decode) (.binary (some .fp32) .real .add a b)=
      a.eval (realISA decode)+b.eval (realISA decode) := rfl
 theorem eval_cast (decode : BitVec 32 → ℝ) (a : Graph) :
    Graph.eval (realISA decode) (.cast (some .fp32) .real .real a)=a.eval (realISA decode) := rfl

theorem lower_real (decode : BitVec 32 → ℝ) (term : Term) :
    (lower term).eval (realISA decode)=term.meaning decode := by
  induction term with
  | bits b => rfl
  | literal x => rfl
  | cast a ih => exact ih
  | load a ih => exact ih
  | mul a b ha hb =>
    change (lower a).eval (realISA decode)*(lower b).eval (realISA decode)=_
    rw [ha,hb]
    rfl
  | dot k a b ha hb =>
    simp only [lower,ieee_real,eval_zero,Term.meaning,ha,hb,zero_add]
  | add a b ha hb =>
    cases b with
    | dot k x y =>
      simp only [lower,ieee_real,eval_zero,zero_add] at hb ⊢
      rw [ha,hb]
      rfl
    | mul b c =>
      change (lower b).eval (realISA decode)*(lower c).eval (realISA decode)+(lower a).eval (realISA decode)=_
      change (lower b).eval (realISA decode)*(lower c).eval (realISA decode)=_ at hb
      rw [hb,ha]
      simp only [Term.meaning,add_comm]
    | bits b =>
      change (lower a).eval (realISA decode)+(lower (.bits b)).eval (realISA decode)=_
      rw [ha,hb]; rfl
    | literal x =>
      change (lower a).eval (realISA decode)+(lower (.literal x)).eval (realISA decode)=_
      rw [ha,hb]; rfl
    | add b c =>
      change (lower a).eval (realISA decode)+(lower (.add b c)).eval (realISA decode)=_
      rw [ha,hb]; rfl
    | cast b =>
      change (lower a).eval (realISA decode)+(lower (.cast b)).eval (realISA decode)=_
      rw [ha,hb]; rfl
    | load b =>
      change (lower a).eval (realISA decode)+(lower (.load b)).eval (realISA decode)=_
      rw [ha,hb]; rfl

def lowerKey (single : Bool) : Term → Graph
  | .cast a => .cast (some .fp32) .real .real (lowerKey single a)
  | .add acc final =>
    if single then
      match acc,final with
      | .add (.literal z) (.dot k a b), .dot l x y =>
        if z=0 then ieeeDot k (fun e => lower (a e)) (fun e => lower (b e)) (lower (.dot l x y))
        else .binary (some .fp32) .real .add (lower acc) (lower final)
      | _,_ => .binary (some .fp32) .real .add (lower acc) (lower final)
    else .binary (some .fp32) .real .add (lower acc) (lower final)
  | a => lower a

theorem lowerKey_real (decode : BitVec 32 → ℝ) (single : Bool) (term : Term) :
    (lowerKey single term).eval (realISA decode)=term.meaning decode := by
  induction term with
  | cast a ih => exact ih
  | add acc final ha hb =>
    simp only [lowerKey]
    split
    · split
      · split
        · simp only [ieee_real,lower_real,Term.meaning]
          simp_all only [zero_add]
          ring
        · simp only [eval_add,lower_real,Term.meaning]
      · simp only [eval_add,lower_real,Term.meaning]
    · simp only [eval_add,lower_real,Term.meaning]
  | bits b => exact lower_real decode _
  | literal x => exact lower_real decode _
  | mul a b ha hb => exact lower_real decode _
  | load a ha => exact lower_real decode _
  | dot k a b ha hb => exact lower_real decode _

#axiomsClean ieee_real
#axiomsClean lower_real
#axiomsClean lowerKey_real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
