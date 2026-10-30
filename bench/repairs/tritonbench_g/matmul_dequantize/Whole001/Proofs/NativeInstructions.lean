import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Kernels
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
open VeriTile.Triton
noncomputable section

/-- Only scalar conversions/arithmetic and one output coordinate of a fixed
SM80 m16n8k16 instruction carry deferred numerical interpretation. -/
structure WordISA where
  literal16 : ℝ → BitVec 16
  literal32 : ℝ → BitVec 32
  narrow : BitVec 32 → BitVec 16
  widen : BitVec 16 → BitVec 32
  add32 : BitVec 32 → BitVec 32 → BitVec 32
  mul32 : BitVec 32 → BitVec 32 → BitVec 32
  mul16 : BitVec 16 → BitVec 16 → BitVec 16
  sub16 : BitVec 16 → BitVec 16 → BitVec 16
  fma16 : BitVec 16 → BitVec 16 → BitVec 16 → BitVec 16
  mma16 : Fin 16 → Fin 8 → (Fin 16 → BitVec 16) → (Fin 16 → BitVec 16) → BitVec 32 → BitVec 32

inductive Graph where
  | bits16 (word : BitVec 16)
  | bits32 (word : BitVec 32)
  | literal16 (value : ℝ)
  | literal32 (value : ℝ)
  | narrow (value : Graph)
  | widen (value : Graph)
  | neg16 (value : Graph)
  | add32 (left right : Graph)
  | mul32 (left right : Graph)
  | mul16 (left right : Graph)
  | sub16 (left right : Graph)
  | fma16 (left right seed : Graph)
  | mma16 (row : Fin 16) (col : Fin 8) (a b : Fin 16 → Graph) (seed : Graph)

def embed16 (word : BitVec 16) : BitVec 32 := BitVec.ofNat 32 word.toNat
def low16 (word : BitVec 32) : BitVec 16 := BitVec.ofNat 16 word.toNat

def Graph.eval (I : WordISA) : Graph → BitVec 32
  | .bits16 w => embed16 w
  | .bits32 w => w
  | .literal16 x => embed16 (I.literal16 x)
  | .literal32 x => I.literal32 x
  | .narrow x => embed16 (I.narrow (x.eval I))
  | .widen x => I.widen (low16 (x.eval I))
  | .neg16 x => embed16 (low16 (x.eval I) ^^^ (0x8000 : BitVec 16))
  | .add32 a b => I.add32 (a.eval I) (b.eval I)
  | .mul32 a b => I.mul32 (a.eval I) (b.eval I)
  | .mul16 a b => embed16 (I.mul16 (low16 (a.eval I)) (low16 (b.eval I)))
  | .sub16 a b => embed16 (I.sub16 (low16 (a.eval I)) (low16 (b.eval I)))
  | .fma16 a b c => embed16 (I.fma16 (low16 (a.eval I)) (low16 (b.eval I)) (low16 (c.eval I)))
  | .mma16 row col a b c =>
      I.mma16 row col (fun k => low16 ((a k).eval I)) (fun k => low16 ((b k).eval I)) (c.eval I)

def dotStep (D row col block : Nat) (a b : Fin D → Graph) (seed : Graph) : Graph :=
  .mma16 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
    (fun lane => if h : block*16+lane.val<D then a ⟨block*16+lane.val,h⟩ else .bits16 0)
    (fun lane => if h : block*16+lane.val<D then b ⟨block*16+lane.val,h⟩ else .bits16 0) seed

def dot (D row col : Nat) (a b : Fin D → Graph) (seed : Graph) : Graph :=
  (List.range ((D+15)/16)).foldl (fun previous block => dotStep D row col block a b previous) seed

theorem dot_feature_covered (D e : Nat) (he : e<D) :
    e/16∈List.range ((D+15)/16) ∧ e%16<16 ∧ e/16*16+e%16=e := by
  refine ⟨List.mem_range.mpr ?_,Nat.mod_lt _ (by decide),by simpa [Nat.mul_comm] using Nat.div_add_mod e 16⟩
  omega

theorem dot_feature_unique (e block lane : Nat) (hl : lane<16) (he : block*16+lane=e) :
    block=e/16 ∧ lane=e%16 := by omega

#axiomsClean dot_feature_covered
#axiomsClean dot_feature_unique
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
