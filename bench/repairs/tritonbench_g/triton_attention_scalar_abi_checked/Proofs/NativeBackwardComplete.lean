import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeBackwardLoop
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000

theorem outer_readonly (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (b : Nat) (r : RegionName) (off : Nat) (hr : r≠"DQ" ∧ r≠"DK" ∧ r≠"DV") :
    backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s b mem r off=mem r off := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro i hi hm;exact Or.inl hr.2.1)]
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro i hi hm;exact Or.inl hr.2.2)]
  exact backwardInner_readonly _ _ _ _ _ _ _ _ hr.1

theorem outer_ready (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head b : Nat) (h : BackwardReady c scale s mem x head) :
    BackwardReady c scale s (backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s b mem) x head := by
  apply h.memory_congr c scale s mem _ x head
  intro r hr o
  apply outer_readonly
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> decide

theorem outer_query (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb : Nat) (h : BackwardReady c scale s mem x head) (hk : kb<c.blocks)
    (initial : Nat → BLane (backwardParameters c scale) → Graph)
    (hq : ∀ b,b<c.blocks → ∀ i,Represents one (mem "DQ" (backwardOffset (backwardParameters c scale) s b i)) (initial b i))
    (b : Nat) (hb : b<c.blocks) (i : BLane (backwardParameters c scale)) :
    Represents one (backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s kb mem "DQ"
      (backwardOffset (backwardParameters c scale) s b i))
      (Target.queryUpdate c one scale x head (b*128+i.1.val) i.2.1.val kb (initial b i)) := by
  have hdiv : (b*128+i.1.val)/128=b := by
    have hi : i.1.val<128:=i.1.isLt
    omega
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;left;simp [backwardParameters])]
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;left;simp [backwardParameters])]
  simp only [Target.queryUpdate,hdiv]
  by_cases he : kb≤b
  · rw [if_pos he]
    exact inner_query c scale one s mem x head kb h hk initial hq _ (by change kb+(c.blocks-kb)≤c.blocks;omega) b
      (by change kb≤b ∧ b<kb+(c.blocks-kb);omega) i
  · rw [if_neg he]
    have hm:=backwardInner_otherBlock (modelScalar one .backward) (backwardParameters c scale) s mem kb
      ((backwardParameters c scale).blocks-kb) b i (Or.inl (by omega))
    change (backwardInnerRounds _ _ _ _ _ _).mem "DQ" _=mem "DQ" _ at hm
    rw [hm]
    exact hq b hb i

theorem pointer_inBounds (c : Shape) (scale : ℝ) (s : State Expr) (r : RegionName) (b : Nat)
    (hb : ∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+b*128+j.val<(backwardParameters c scale).rows)
    (i : BLane (backwardParameters c scale)) :
    (backwardPointer (backwardParameters c scale) s r b).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true := by
  have hs : backwardHeadRow (backwardParameters c scale) s+b*(backwardParameters c scale).blockM+i.1.val<
      (backwardParameters c scale).rows:=hb i.1
  simp [backwardPointer,TileShape.blockPtr_inBounds_2d_offsets_index,hs,i.2.1.isLt]

theorem outer_key (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb : Nat) (h : BackwardReady c scale s mem x head) (hk : kb<c.blocks)
    (hb : ∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+kb*128+j.val<(backwardParameters c scale).rows)
    (i : BLane (backwardParameters c scale)) :
    Represents one (backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s kb mem "DK"
      (backwardOffset (backwardParameters c scale) s kb i))
      (Target.half (Target.keyPrefix c one scale x head kb i.1.val i.2.1.val (c.blocks-kb))) := by
  change Represents one (backwardOuterAdvance _ _ _ _ _ (backwardParameters c scale).dk _) _
  unfold backwardOuterAdvance
  simp only [pointer_inBounds c scale s _ kb hb,ite_true]
  rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective _ _ _)]
  refine ⟨_,rfl,?_⟩
  intro u
  change Target.half (lowerScalar one .backward
    ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb (c.blocks-kb)).dk i u))=_
  rw [(inner_prefixes c scale one s mem x head kb h hk _ (by omega)).1 i u]

theorem outer_value (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (x : Target.BackwardInputs) (head kb : Nat) (h : BackwardReady c scale s mem x head) (hk : kb<c.blocks)
    (hb : ∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+kb*128+j.val<(backwardParameters c scale).rows)
    (i : BLane (backwardParameters c scale)) :
    Represents one (backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s kb mem "DV"
      (backwardOffset (backwardParameters c scale) s kb i))
      (Target.half (Target.valuePrefix c one scale x head kb i.1.val i.2.1.val (c.blocks-kb))) := by
  change Represents one (backwardOuterAdvance _ _ _ _ _ (backwardParameters c scale).dv _) _
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame _ _ _ _ _ _ _ _ (by intro j hj hm;left;simp [backwardParameters])]
  simp only [pointer_inBounds c scale s _ kb hb,ite_true]
  rw [memoryFold_readback _ _ _ _ i (backwardOffset_injective _ _ _)]
  refine ⟨_,rfl,?_⟩
  intro u
  change Target.half (lowerScalar one .backward
    ((backwardInnerRounds (modelScalar one .backward) (backwardParameters c scale) s mem kb (c.blocks-kb)).dv i u))=_
  rw [(inner_prefixes c scale one s mem x head kb h hk _ (by omega)).2 i u]

theorem outer_otherKey (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr) (mem : Source.Memory Expr)
    (kb b : Nat) (he : b≠kb) (r : RegionName) (hr : r="DK" ∨ r="DV") (i : BLane (backwardParameters c scale)) :
    backwardOuterAdvance (modelScalar one .backward) (backwardParameters c scale) s kb mem r
      (backwardOffset (backwardParameters c scale) s b i)=mem r (backwardOffset (backwardParameters c scale) s b i) := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame]
  · rw [memoryMaskedFold_frame]
    · apply backwardInner_readonly
      rcases hr with rfl|rfl <;> simp [backwardParameters]
    · intro j hj hm
      exact Or.inr (fun hEq=>he (backwardOffset_inj _ _ b kb i j hEq).1)
  · intro j hj hm
    exact Or.inr (fun hEq=>he (backwardOffset_inj _ _ b kb i j hEq).1)

theorem rounds_ready (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.BackwardInputs) (head : Nat) (h : BackwardReady c scale s s.mem x head) (n : Nat) :
    BackwardReady c scale s (backwardRounds (modelScalar one .backward) (backwardParameters c scale) s n) x head := by
  induction n with
  | zero=>exact h
  | succ n ih=>exact outer_ready c scale one s _ x head n ih

theorem rounds_query (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.BackwardInputs) (head : Nat) (h : BackwardReady c scale s s.mem x head)
    (hq : ∀ b,b<c.blocks → ∀ i,Represents one (s.mem "DQ" (backwardOffset (backwardParameters c scale) s b i)) (.word 0))
    (n : Nat) (hn : n≤c.blocks) (b : Nat) (hb : b<c.blocks) (i : BLane (backwardParameters c scale)) :
    Represents one (backwardRounds (modelScalar one .backward) (backwardParameters c scale) s n "DQ"
      (backwardOffset (backwardParameters c scale) s b i))
      (Target.query c one scale x head (b*128+i.1.val) i.2.1.val n) := by
  induction n generalizing b i with
  | zero=>exact hq b hb i
  | succ n ih=>
    exact outer_query c scale one s _ x head n (rounds_ready c scale one s x head h n) (by omega)
      (fun b i=>Target.query c one scale x head (b*128+i.1.val) i.2.1.val n)
      (by intro b hb i;exact ih (by omega) b hb i) b hb i

theorem rounds_kv (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.BackwardInputs) (head : Nat) (h : BackwardReady c scale s s.mem x head)
    (hrow : ∀ b<c.blocks,∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+b*128+j.val<(backwardParameters c scale).rows)
    (n : Nat) (hn : n≤c.blocks) (b : Nat) (hb : b<n) (i : BLane (backwardParameters c scale)) :
    Represents one (backwardRounds (modelScalar one .backward) (backwardParameters c scale) s n "DK"
      (backwardOffset (backwardParameters c scale) s b i))
      (Target.half (Target.keyPrefix c one scale x head b i.1.val i.2.1.val (c.blocks-b))) ∧
    Represents one (backwardRounds (modelScalar one .backward) (backwardParameters c scale) s n "DV"
      (backwardOffset (backwardParameters c scale) s b i))
      (Target.half (Target.valuePrefix c one scale x head b i.1.val i.2.1.val (c.blocks-b))) := by
  induction n with
  | zero=>omega
  | succ n ih=>
    by_cases he : b=n
    · subst b
      exact ⟨outer_key c scale one s _ x head n (rounds_ready c scale one s x head h n) (by omega) (hrow n (by omega)) i,
        outer_value c scale one s _ x head n (rounds_ready c scale one s x head h n) (by omega) (hrow n (by omega)) i⟩
    · simp only [backwardRounds]
      rw [outer_otherKey c scale one s _ n b he "DK" (Or.inl rfl) i,
        outer_otherKey c scale one s _ n b he "DV" (Or.inr rfl) i]
      exact ih (by omega) (by omega)

theorem keyPrefix_fold (c : Shape) (one : Bool) (scale : ℝ) (x : Target.BackwardInputs) (h kb row d n : Nat) :
    Target.keyPrefix c one scale x h kb row d n=
      (List.range n).foldl (fun seed t=>Target.keyStep c one scale x h (kb*128+row) d (kb+t) seed) (.word 0) := by
  induction n with
  | zero=>rfl
  | succ n ih=>simp only [Target.keyPrefix,List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,ih]

theorem valuePrefix_fold (c : Shape) (one : Bool) (scale : ℝ) (x : Target.BackwardInputs) (h kb row d n : Nat) :
    Target.valuePrefix c one scale x h kb row d n=
      (List.range n).foldl (fun seed t=>Target.valueStep c one scale x h (kb*128+row) d (kb+t) seed) (.word 0) := by
  induction n with
  | zero=>rfl
  | succ n ih=>simp only [Target.valuePrefix,List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,ih]

theorem keyPrefix_target (c : Shape) (one : Bool) (scale : ℝ) (x : Target.BackwardInputs) (h kb row d : Nat) (hr : row<128) :
    Target.half (Target.keyPrefix c one scale x h kb row d (c.blocks-kb))=Target.key c one scale x h (kb*128+row) d := by
  have hd : (kb*128+row)/128=kb:=by omega
  simp only [Target.key,hd,keyPrefix_fold,Target.keyStep]

theorem valuePrefix_target (c : Shape) (one : Bool) (scale : ℝ) (x : Target.BackwardInputs) (h kb row d : Nat) (hr : row<128) :
    Target.half (Target.valuePrefix c one scale x h kb row d (c.blocks-kb))=Target.value c one scale x h (kb*128+row) d := by
  have hd : (kb*128+row)/128=kb:=by omega
  simp only [Target.value,hd,valuePrefix_fold,Target.valueStep]

/-- The full shared backward CTA computes the independent native graph target,
including the actual DQ memory seed and every DK/DV accumulator update. -/
theorem backward_execution_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.BackwardInputs) (head : Nat) (h : BackwardReady c scale s s.mem x head)
    (hrow : ∀ b<c.blocks,∀ j : Fin 128,backwardHeadRow (backwardParameters c scale) s+b*128+j.val<(backwardParameters c scale).rows)
    (hq : ∀ b,b<c.blocks → ∀ i,Represents one (s.mem "DQ" (backwardOffset (backwardParameters c scale) s b i)) (.word 0)) :
    ∃ final,FP.Structural.exec (backend (algebraScalar one .backward) (modelScalar one .backward)) (backwardKernel c scale) s=some final ∧
      (∀ b,b<c.blocks → ∀ i : BLane (backwardParameters c scale),
        Represents one (final.mem "DQ" (backwardOffset (backwardParameters c scale) s b i))
          (Target.query c one scale x head (b*128+i.1.val) i.2.1.val c.blocks) ∧
        Represents one (final.mem "DK" (backwardOffset (backwardParameters c scale) s b i))
          (Target.key c one scale x head (b*128+i.1.val) i.2.1.val) ∧
        Represents one (final.mem "DV" (backwardOffset (backwardParameters c scale) s b i))
          (Target.value c one scale x head (b*128+i.1.val) i.2.1.val)) ∧
      (∀ r o,¬backwardWritten (backwardParameters c scale) s r o → final.mem r o=s.mem r o) := by
  obtain ⟨final,he,hm,_,hf⟩:=backwardCorrectness (algebraScalar one .backward) (modelScalar one .backward)
    (backwardParameters c scale) s (by change 0<128;decide)
  refine ⟨final,he,?_,hf⟩
  intro b hb i
  rw [hm]
  have hk:=rounds_kv c scale one s x head h hrow c.blocks (by omega) b hb i
  rw [keyPrefix_target c one scale x head b i.1.val i.2.1.val i.1.isLt,
    valuePrefix_target c one scale x head b i.1.val i.2.1.val i.1.isLt] at hk
  exact ⟨rounds_query c scale one s x head h hq c.blocks (by omega) b hb i,hk⟩

#axiomsClean backward_execution_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
