import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Dimension
import bench.repairs.tritonbench_g.triton_argmax.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 2000000

/-- A scalar scratch boundary: exactly one FP32 conversion, the decoded stored
ordering key, and the source padding word. It contains no index, reduction,
loop, memory, or final-result function. -/
structure Scratch (d : MidFormat) where
  cast : Word → MidWord d
  key : MidWord d → WithTop Int
  padding : MidWord d

def Scratch.scalars {d : MidFormat} (scratch : Scratch d) : Scalars d :=
  ⟨greater,equal,0xff800000,scratch.cast,
   fun x y => orderedGt (scratch.key x) (scratch.key y),
   fun x y => orderedEq (scratch.key x) (scratch.key y),scratch.padding⟩

/-- Integer/bool conversions retain the selected repair's complete
NaN-to-zero, clamp, truncation, typed-store and decoded-load macro. -/
def integerScratch (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind) :
    Scratch (Formats.format kind) where
  cast := (ScalarBinding.scalars kind p).storeCast
  key := fun x => (Formats.decodeStored kind x : WithTop Int)
  padding := (ScalarBinding.scalars kind p).negInfMid

/-- Concrete IEEE storage classification for all four requested floating
scratch formats. NaNs share top priority; both zero signs have one key. -/
def fractionBits : RealScratch.FloatingKind → Nat
  | .fp16 => 10
  | .bf16 => 7
  | .fp32 => 23
  | .fp64 => 52

def floatMagnitude (kind : RealScratch.FloatingKind) (x : MidWord kind.format) : Nat :=
  x.toNat % 2^(8*kind.format.bytes-1)

def floatNaN (kind : RealScratch.FloatingKind) (x : MidWord kind.format) : Bool :=
  decide (2^(8*kind.format.bytes-1)-2^(fractionBits kind) < floatMagnitude kind x)

def floatKey (kind : RealScratch.FloatingKind) (x : MidWord kind.format) : WithTop Int :=
  if floatNaN kind x then ⊤ else
    if floatMagnitude kind x=0 then (Int.ofNat (2^(8*kind.format.bytes-1)-1) : WithTop Int)
    else if x.toNat<2^(8*kind.format.bytes-1) then
      (Int.ofNat (x.toNat+2^(8*kind.format.bytes-1)) : WithTop Int)
    else (Int.ofNat (2^(8*kind.format.bytes)-1-x.toNat) : WithTop Int)

def floatPadding (kind : RealScratch.FloatingKind) : MidWord kind.format :=
  BitVec.ofNat (8*kind.format.bytes) (2^(8*kind.format.bytes)-2^(fractionBits kind))

/-- The conversion is a local native cast primitive. Its result is retained at
the exact requested width and ranked after decoding; FP16/BF16 rounding is
never replaced by the original FP32 ranking. -/
def floatingScratch (kind : RealScratch.FloatingKind) (cast : Word → MidWord kind.format) :
    Scratch kind.format := ⟨cast,floatKey kind,floatPadding kind⟩

def firstReference (c : GlobalConfig) (positive : 0<c.block) (input : Nat → Word) (pid : Nat) :
    Fin c.block := reference c.block positive (c.loaded input 0xff800000 pid)

def firstReferenceIndex (c : GlobalConfig) (positive : 0<c.block) (input : Nat → Word)
    (pid : Nat) : Nat := pid*c.block+(firstReference c positive input pid).val

def firstReferenceValue {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (input : Nat → Word) (pid : Nat) : MidWord d :=
  scratch.cast (c.loaded input 0xff800000 pid (firstReference c positive input pid))

/-- Input-and-request-only staged target: canonical first-index FP32 ranking,
requested scalar conversion, canonical decoded scratch ranking, then the
original stage-one index. No implementation reduction tree occurs here. -/
def stagedKeys {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (input : Nat → Word) (lane : Fin c.midBlock) : WithTop Int :=
  scratch.key (if lane.val<c.blocks then firstReferenceValue scratch c positive input lane.val
    else scratch.padding)

def secondReference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (midPositive : 0<c.midBlock) (input : Nat → Word) : Fin c.midBlock :=
  (Math.enumeration c.midBlock midPositive).pick orderedGt orderedEq (stagedKeys scratch c positive input)

def globalReference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (midPositive : 0<c.midBlock) (input : Nat → Word) : Nat :=
  let selected := secondReference scratch c positive midPositive input
  if selected.val<c.blocks then firstReferenceIndex c positive input selected.val else 0

theorem first_reference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (input : Nat → Word) (pid : Nat)
    (complete : (c.firstTree pid).Complete) :
    c.selected scratch.scalars.gt32 scratch.scalars.eq32 input scratch.scalars.negInf32 pid=
      firstReference c positive input pid :=
  pick_reference positive _ _ complete

theorem first_index_reference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (input : Nat → Word) (pid : Nat)
    (complete : (c.firstTree pid).Complete) :
    stage1Index scratch.scalars c input pid=firstReferenceIndex c positive input pid := by
  simp only [stage1Index,first_reference scratch c positive input pid complete,firstReferenceIndex,Nat.add_comm]

theorem first_value_reference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (input : Nat → Word) (pid : Nat)
    (complete : (c.firstTree pid).Complete) :
    stage1Value scratch.scalars c input pid=firstReferenceValue scratch c positive input pid := by
  simp only [stage1Value,first_reference scratch c positive input pid complete]
  rfl

theorem pick_key {α β : Type} (key : α → β) [LinearOrder β] {n : Nat}
    (input : Fin n → α) (tree : Tree n) :
    tree.pick (fun x y => orderedGt (key x) (key y)) (fun x y => orderedEq (key x) (key y)) input=
      tree.pick orderedGt orderedEq (fun i => key (input i)) := by
  induction tree with
  | leaf => rfl
  | fork a b ha hb => simp only [Tree.pick,ha,hb,leftWins]; rfl

theorem staged_maximum {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (midPositive : 0<c.midBlock) (input : Nat → Word) :
    IsLeastMaximum (stagedKeys scratch c positive input)
      (secondReference scratch c positive midPositive input) :=
  mathematical_maximum _ _ (Math.enumeration_complete _ _)

theorem second_reference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (midPositive : 0<c.midBlock) (input : Nat → Word)
    (firstComplete : ∀ pid,pid<c.blocks → (c.firstTree pid).Complete)
    (secondComplete : c.secondTree.Complete) :
    stage2Selected scratch.scalars c (stage1Value scratch.scalars c input)=
      secondReference scratch c positive midPositive input := by
  have keys : (fun lane => scratch.key (stage2Loaded scratch.scalars c
      (stage1Value scratch.scalars c input) lane))=stagedKeys scratch c positive input := by
    funext lane
    dsimp only [stage2Loaded,stagedKeys]
    by_cases active : lane.val<c.blocks
    · simp only [if_pos active,first_value_reference scratch c positive input lane.val (firstComplete _ active)]
    · simp only [if_neg active,Scratch.scalars]
  unfold stage2Selected
  change c.secondTree.pick (fun x y => orderedGt (scratch.key x) (scratch.key y))
    (fun x y => orderedEq (scratch.key x) (scratch.key y)) _=_
  rw [pick_key,keys]
  exact least_maximum_unique _ _ _ (mathematical_maximum _ _ secondComplete)
    (staged_maximum scratch c positive midPositive input)

theorem global_reference {d : MidFormat} (scratch : Scratch d) (c : GlobalConfig)
    (positive : 0<c.block) (midPositive : 0<c.midBlock) (input : Nat → Word)
    (firstComplete : ∀ pid,pid<c.blocks → (c.firstTree pid).Complete)
    (secondComplete : c.secondTree.Complete) :
    Source.globalIndex scratch.scalars c input=globalReference scratch c positive midPositive input := by
  unfold Source.globalIndex globalReference
  rw [second_reference scratch c positive midPositive input firstComplete secondComplete]
  dsimp only
  split
  · exact first_index_reference scratch c positive input _ (firstComplete _ (by assumption))
  · rfl

#axiomsClean staged_maximum
#axiomsClean first_index_reference
#axiomsClean first_value_reference
#axiomsClean global_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst
