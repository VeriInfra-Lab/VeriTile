import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeBackwardDots
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardMemory
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

namespace Target
def keyStep (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h j d qb : Nat) (seed : Graph) : Graph :=
  dot (TritonAttention.Whole001.Native.width c.dim) 128
    (pad 128 (fun r=>half (backwardDS c one scale x h (qb*128+r) j)))
    (pad 128 (fun r=>x.q h (qb*128+r) d)) Bool.true seed
def valueStep (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h j d qb : Nat) (seed : Graph) : Graph :=
  dot (TritonAttention.Whole001.Native.width c.dim) 128
    (pad 128 (fun r=>half (backwardProbability c one scale x h (qb*128+r) j)))
    (pad 128 (fun r=>x.gradient h (qb*128+r) d)) Bool.true seed
def keyPrefix (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h kb row d : Nat) : Nat → Graph
  | 0=>.word 0
  | n+1=>keyStep c one scale x h (kb*128+row) d (kb+n) (keyPrefix c one scale x h kb row d n)
def valuePrefix (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h kb row d : Nat) : Nat → Graph
  | 0=>.word 0
  | n+1=>valueStep c one scale x h (kb*128+row) d (kb+n) (valuePrefix c one scale x h kb row d n)
def queryStep (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h row d kb : Nat) (seed : Graph) : Graph :=
  dot (TritonAttention.Whole001.Native.width c.dim) 128
    (pad 128 (fun j=>half (backwardDS c one scale x h row (kb*128+j))))
    (pad 128 (fun j=>x.k h (kb*128+j) d)) Bool.true seed
end Target

theorem BackwardReady.memory_congr (c : Shape) (scale : ℝ) (s : State Expr) (mem mem' : Source.Memory Expr)
    (x : Target.BackwardInputs) (head : Nat) (h : BackwardReady c scale s mem x head)
    (hm : ∀ r∈(["Q","K","V","NewDO","M","Delta"] : List RegionName),∀ o,mem' r o=mem r o) :
    BackwardReady c scale s mem' x head := by
  have readEq (r : RegionName) (hr : r∈(["Q","K","V","NewDO","M","Delta"] : List RegionName)) (b : Nat)
      (i : BLane (backwardParameters c scale)) :
      backwardRead (backwardParameters c scale) s mem' r b i=backwardRead (backwardParameters c scale) s mem r b i := by
    unfold backwardRead
    rw [hm r hr]
  constructor
  · intro b hb i;rw [readEq "Q" (by simp)];exact h.q b hb i
  · intro b hb i;rw [readEq "K" (by simp)];exact h.k b hb i
  · intro b hb i;rw [readEq "V" (by simp)];exact h.v b hb i
  · intro b hb i;rw [readEq "NewDO" (by simp)];exact h.grad b hb i
  · intro b hb i;rw [hm "M" (by simp)];exact h.maximum b hb i
  · intro b hb i;rw [hm "Delta" (by simp)];exact h.delta b hb i

theorem inner_ready (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb n : Nat) (h : BackwardReady c scale s mem x head) :
    BackwardReady c scale s (backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).mem x head := by
  apply h.memory_congr c scale s mem _ x head
  intro r hr o
  apply backwardInner_readonly
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> simp [backwardParameters]

theorem inner_prefixes (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb : Nat) (h : BackwardReady c scale s mem x head) (hkb : kb<c.blocks)
    (n : Nat) (hn : kb+n≤c.blocks) :
    (∀ i : BLane (backwardParameters c scale),∀ u,lowerScalar one .backward
      ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).dk i u)=
        Target.keyPrefix c one scale x head kb i.1.val i.2.1.val n) ∧
    (∀ i : BLane (backwardParameters c scale),∀ u,lowerScalar one .backward
      ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).dv i u)=
        Target.valuePrefix c one scale x head kb i.1.val i.2.1.val n) := by
  induction n with
  | zero=>constructor <;> intro i u <;>
      simp [backwardInnerRounds,backwardInnerInitial,modelScalar,algebraScalar,algebra,literal,lowerScalar,Target.keyPrefix,Target.valuePrefix]
  | succ n ih=>
    have prev:=ih (by omega)
    have hr:=inner_ready c scale one s mem x head kb n h
    constructor
    · intro i u
      change lowerScalar one .backward (.add
        ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).dk i u)
        (backwardKeyDot (modelScalar one .backward) (backwardParameters c scale) s _
          (backwardRead (backwardParameters c scale) s mem "K" kb) (backwardRead (backwardParameters c scale) s mem "V" kb)
          kb (kb+n) i u))=_
      rw [backwardKeyDot_native c scale one s _ x head kb (kb+n) hr (by omega) _ _ (h.k kb hkb) (h.v kb hkb),lower_seeded_dot,prev.1 i u]
      rfl
    · intro i u
      change lowerScalar one .backward (.add
        ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).dv i u)
        (backwardValueDot (modelScalar one .backward) (backwardParameters c scale) s _
          (backwardRead (backwardParameters c scale) s mem "K" kb) kb (kb+n) i u))=_
      rw [backwardValueDot_native c scale one s _ x head kb (kb+n) hr (by omega) _ (h.k kb hkb),lower_seeded_dot,prev.2 i u]
      rfl

def Represents (one : Bool) (cell : Cell Expr) (g : Graph) : Prop :=
  ∃ e : Expr,cell=.mk .fp32 e ∧ ∀ u,lowerScalar one .backward (e u)=g

theorem Represents.read (one : Bool) (cell : Cell Expr) (g : Graph) (h : Represents one cell g) (u : Nat) :
    lowerScalar one .backward (cell.read .fp32 u)=g := by
  obtain ⟨e,he,hg⟩:=h
  rw [he]
  exact hg u

theorem inner_query (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb : Nat) (h : BackwardReady c scale s mem x head) (hkb : kb<c.blocks)
    (initial : Nat → BLane (backwardParameters c scale) → Graph)
    (hq : ∀ b,b<c.blocks → ∀ i,Represents one (mem "DQ" (backwardOffset (backwardParameters c scale) s b i)) (initial b i))
    (n : Nat) (hn : kb+n≤c.blocks) (b : Nat) (hbn : kb≤b ∧ b<kb+n) (i : BLane (backwardParameters c scale)) :
    Represents one ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).mem "DQ"
      (backwardOffset (backwardParameters c scale) s b i))
      (Target.queryStep c one scale x head (b*128+i.1.val) i.2.1.val kb (initial b i)) := by
  induction n with
  | zero=>omega
  | succ n ih=>
    change Represents one (((TileShape.allIndices [(backwardParameters c scale).blockM,(backwardParameters c scale).head]).foldl
      (fun m j=>writeMemory m (backwardParameters c scale).dq
      (backwardOffset (backwardParameters c scale) s (kb+n) j) _) _) (backwardParameters c scale).dq
      (backwardOffset (backwardParameters c scale) s b i)) _
    by_cases he : b=kb+n
    · subst b
      rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective _ _ _)]
      refine ⟨_,rfl,?_⟩
      intro u
      change lowerScalar one .backward (.add
        (((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb n).mem "DQ"
          (backwardOffset (backwardParameters c scale) s (kb+n) i)).read .fp32 u)
        (backwardQueryDot (modelScalar one .backward) (backwardParameters c scale) s _
          (backwardRead (backwardParameters c scale) s mem "K" kb) (backwardRead (backwardParameters c scale) s mem "V" kb)
          kb (kb+n) i u))=_
      rw [backwardQueryDot_native c scale one s _ x head kb (kb+n) (inner_ready c scale one s mem x head kb n h)
        (by omega) _ _ (h.k kb hkb) (h.v kb hkb),lower_seeded_dot]
      have hmem:=backwardInner_otherBlock (modelScalar one .backward) (backwardParameters c scale) s mem kb n
        (kb+n) i (Or.inr (by omega))
      change (backwardInnerRounds _ _ _ _ _ _).mem "DQ" _=mem "DQ" _ at hmem
      rw [hmem,Represents.read one _ _ (hq _ (by omega) i) u]
      rfl
    · rw [memoryFold_frame]
      · exact ih (by omega) (by omega)
      · intro j hj
        exact Or.inr (fun hEq=>he (backwardOffset_inj _ _ b (kb+n) i j hEq).1)

#axiomsClean inner_prefixes
#axiomsClean inner_query
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
