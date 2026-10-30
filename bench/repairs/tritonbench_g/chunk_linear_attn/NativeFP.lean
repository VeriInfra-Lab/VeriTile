import bench.repairs.tritonbench_g.chunk_linear_attn.NativePorts
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

/-- Only scalar FP32 primitives. A profile is never a tensor/dot callback. -/
structure FP32Profile where
  literal : ℝ → BitVec 32
  add : BitVec 32 → BitVec 32 → BitVec 32
  mul : BitVec 32 → BitVec 32 → BitVec 32
  cast : BitVec 32 → BitVec 32
  fma : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32

def FP32Profile.isa (p : FP32Profile) : ISA (BitVec 32) where
  bits := id
  literal := fun _ _ => p.literal
  binary := fun _ _ op a b => match op with | .add => p.add a b | .mul => p.mul a b | _ => a
  unary := fun _ _ a => a
  cast := fun _ _ _ => p.cast
  fma := p.fma
  mmaTF32 := fun _ _ _ _ c => c

structure PrimitiveLeaves (I : ISA (BitVec 32)) (p : FP32Profile) : Prop where
  bits : ∀ b, I.bits b=b
  literal : ∀ x, I.literal (some .fp32) .real x=p.literal x
  add : ∀ a b, I.binary (some .fp32) .real .add a b=p.add a b
  mul : ∀ a b, I.binary (some .fp32) .real .mul a b=p.mul a b
  cast : ∀ a, I.cast (some .fp32) .real .real a=p.cast a
  fma : ∀ a b c, I.fma a b c=p.fma a b c

inductive ScalarGraph : Graph → Prop where
  | bits (b) : ScalarGraph (.bits b)
  | literal (x) : ScalarGraph (.literal (some .fp32) .real x)
  | add {a b} : ScalarGraph a → ScalarGraph b → ScalarGraph (.binary (some .fp32) .real .add a b)
  | mul {a b} : ScalarGraph a → ScalarGraph b → ScalarGraph (.binary (some .fp32) .real .mul a b)
  | cast {a} : ScalarGraph a → ScalarGraph (.cast (some .fp32) .real .real a)
  | fma {a b c} : ScalarGraph a → ScalarGraph b → ScalarGraph c → ScalarGraph (.fma a b c)
  | move {a} (r) : ScalarGraph a → ScalarGraph (.move r a)

theorem ieee_scalar (k : Nat) (a b : Fin k → Graph) (seed : Graph)
    (ha : ∀ e, ScalarGraph (a e)) (hb : ∀ e, ScalarGraph (b e)) (hs : ScalarGraph seed) :
    ScalarGraph (ieeeDot k a b seed) := by
  have aux : ∀ (xs : List (Fin k)) (s : Graph), ScalarGraph s →
      ScalarGraph (xs.foldl (fun acc e => .fma (a e) (b e) acc) s) := by
    intro xs
    induction xs with
    | nil => intro s hs; exact hs
    | cons e xs ih => intro s hs; exact ih _ (.fma (ha e) (hb e) hs)
  exact aux _ seed hs

theorem lower_scalar : (t : Term) → ScalarGraph (lower t)
  | .bits b => .bits b
  | .literal x => .literal x
  | .cast a => .cast (lower_scalar a)
  | .load a => .move _ (lower_scalar a)
  | .mul a b => .mul (lower_scalar a) (lower_scalar b)
  | .dot k a b => ieee_scalar k _ _ _ (fun e => lower_scalar (a e)) (fun e => lower_scalar (b e)) (.literal 0)
  | .add a (.dot k x y) => ieee_scalar k _ _ _ (fun e => lower_scalar (x e)) (fun e => lower_scalar (y e)) (lower_scalar a)
  | .add a (.mul b c) => .fma (lower_scalar b) (lower_scalar c) (lower_scalar a)
  | .add a (.bits b) => .add (lower_scalar a) (.bits b)
  | .add a (.literal x) => .add (lower_scalar a) (.literal x)
  | .add a (.add b c) => .add (lower_scalar a) (lower_scalar (.add b c))
  | .add a (.cast b) => .add (lower_scalar a) (.cast (lower_scalar b))
  | .add a (.load b) => .add (lower_scalar a) (.move _ (lower_scalar b))

theorem lowerKey_scalar (single : Bool) (t : Term) : ScalarGraph (lowerKey single t) := by
  induction t with
  | cast a ha => exact .cast ha
  | add acc final ha hb =>
    simp only [lowerKey]
    split
    · split
      · split
        · exact ieee_scalar _ _ _ _ (fun e => lower_scalar _) (fun e => lower_scalar _) (lower_scalar _)
        · exact .add (lower_scalar _) (lower_scalar _)
      · exact .add (lower_scalar _) (lower_scalar _)
    · exact .add (lower_scalar _) (lower_scalar _)
  | bits b => exact lower_scalar _
  | literal x => exact lower_scalar _
  | mul a b ha hb => exact lower_scalar _
  | load a ha => exact lower_scalar _
  | dot k a b ha hb => exact lower_scalar _

theorem graph_scalar (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    ScalarGraph (graph c bits r o) := by
  unfold graph lowerAt
  split
  · exact lowerKey_scalar _ _
  · exact lower_scalar _

theorem scalar_transport (I : ISA (BitVec 32)) (p : FP32Profile) (h : PrimitiveLeaves I p)
    (g : Graph) (hg : ScalarGraph g) : g.eval I=g.eval p.isa := by
  induction hg with
  | bits b => exact h.bits b
  | literal x => exact h.literal x
  | add ha hb ia ib => simpa only [Graph.eval,h.add,ia,ib,FP32Profile.isa]
  | mul ha hb ia ib => simpa only [Graph.eval,h.mul,ia,ib,FP32Profile.isa]
  | cast ha ia => simpa only [Graph.eval,h.cast,ia,FP32Profile.isa]
  | fma ha hb hc ia ib ic => simpa only [Graph.eval,h.fma,ia,ib,ic,FP32Profile.isa]
  | move r ha ia => exact ia

/-- Satisfiable native-word correspondence. This claims no exact FP32-to-real
law and no accuracy bound; Real evaluation is proved separately by graph_real. -/
theorem graph_fp (I : ISA (BitVec 32)) (p : FP32Profile) (h : PrimitiveLeaves I p)
    (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    (graph c bits r o).eval I=(graph c bits r o).eval p.isa :=
  scalar_transport I p h _ (graph_scalar c bits r o)

#axiomsClean lower_scalar
#axiomsClean lowerKey_scalar
#axiomsClean graph_scalar
#axiomsClean graph_fp
#specNonCircular PrimitiveLeaves avoiding [graph,rawMemory,completeMemory,Real.Call.target]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
