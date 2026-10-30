import bench.tritonbench_g.triton_argmax.Proofs.TypedScratch
import bench.tritonbench_g.triton_argmax.Proofs.TypedInteger
import bench.tritonbench_g.triton_argmax.Proofs.Request

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedScalarBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section

def Reachable (kind : RealScratch.FloatingKind) (cast : Word → MidWord kind.format)
    (word : MidWord kind.format) : Prop := (∃ raw,word=cast raw) ∨ word=floatPadding kind

/-- Local scalar representation facts for floating scratch. The only decode
operation maps one stored scalar to an exactly equivalent FP32 comparison
encoding (FP64 scratch values are FP32-origin values). No reduction, selected
index, graph result or frame is a field. -/
structure FloatCodec (kind : RealScratch.FloatingKind) (round : TypedScratch.Rounding)
    (cast : Word → MidWord kind.format) where
  decode : MidWord kind.format → Word
  castValue : ∀ word,decode (cast word)=round.cast kind word
  padding : decode (floatPadding kind)=0xff800000
  greater : ∀ x y,Reachable kind cast x → Reachable kind cast y → TritonArgmaxNanFirst.greater (decode x) (decode y)=
    orderedGt (floatKey kind x) (floatKey kind y)
  equal : ∀ x y,Reachable kind cast x → Reachable kind cast y → TritonArgmaxNanFirst.equal (decode x) (decode y)=
    orderedEq (floatKey kind x) (floatKey kind y)

theorem tree_decode {n : Nat} (kind : RealScratch.FloatingKind) (round : TypedScratch.Rounding)
    (cast : Word → MidWord kind.format) (codec : FloatCodec kind round cast)
    (input : Fin n → MidWord kind.format) (tree : Tree n)
    (reachable : ∀ i,Reachable kind cast (input i)) :
    tree.pick greater equal (fun i => codec.decode (input i))=
      tree.pick (floatingScratch kind cast).scalars.gtMid (floatingScratch kind cast).scalars.eqMid input := by
  induction tree with
  | leaf => rfl
  | fork a b ha hb =>
    simp only [Tree.pick,ha,hb,leftWins,codec.greater _ _ (reachable _) (reachable _),
      codec.equal _ _ (reachable _) (reachable _),floatingScratch,Scratch.scalars]
    rfl

theorem floating_selected (kind : RealScratch.FloatingKind) (round : TypedScratch.Rounding)
    (cast : Word → MidWord kind.format) (codec : FloatCodec kind round cast)
    (c : GlobalConfig) (positive : 0<c.midBlock) (complete : c.secondTree.Complete)
    (s : State Word) (V : RegionName) (mid : Nat → MidWord kind.format)
    (reachable : ∀ i,i<c.blocks → Reachable kind cast (mid i))
    (reads : ∀ i,i<c.blocks →
      toFloat (TypedScratch.channel kind) ((s.mem V i).read (TypedScratch.channel kind).toTileDType)=codec.decode (mid i)) :
    TypedScratch.secondIndex kind s V c.blocks c.midBlock positive=
      (stage2Selected (floatingScratch kind cast).scalars c mid).val := by
  unfold TypedScratch.secondIndex
  rw [TypedReduction.selected_eq positive _ c.secondTree complete]
  have values : (fun lane : Fin c.midBlock =>
      if lane.val<c.blocks then toFloat (TypedScratch.channel kind)
        ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType) else 0xff800000)=
      fun lane => codec.decode (stage2Loaded (floatingScratch kind cast).scalars c mid lane) := by
    funext lane
    by_cases active : lane.val<c.blocks
    · simp only [if_pos active,reads lane.val active,stage2Loaded,floatingScratch,Scratch.scalars]
    · simp only [if_neg active,stage2Loaded,floatingScratch,Scratch.scalars]
      exact codec.padding.symm
  rw [values,tree_decode kind round cast codec _ _]
  · rfl
  · intro lane
    unfold stage2Loaded
    split
    · exact reachable lane.val (by assumption)
    · exact Or.inr rfl

theorem integer_selected (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (c : GlobalConfig) (positive : 0<c.midBlock) (complete : c.secondTree.Complete)
    (s : State Word) (V : Region .int) (mid : Nat → MidWord (Formats.format kind))
    (reads : ∀ i,i<c.blocks →
      (s.mem (Region.cast V) i).read .int=Formats.decodeStored kind (mid i)) :
    TypedInteger.selected s V c.blocks c.midBlock
      (Formats.decodeStored kind (integerScratch kind p).padding)=
      (stage2Selected (integerScratch kind p).scalars c mid).val := by
  rw [TypedInteger.selected_tree s V c.blocks c.midBlock _ positive c.secondTree complete]
  have values : (fun lane : Fin c.midBlock => if lane.val<c.blocks then
      (s.mem (Region.cast V) lane.val).read .int else Formats.decodeStored kind (integerScratch kind p).padding)=
      fun lane => Formats.decodeStored kind (stage2Loaded (integerScratch kind p).scalars c mid lane) := by
    funext lane
    by_cases active : lane.val<c.blocks
    · simp [stage2Loaded,active,reads lane.val active,Scratch.scalars]
    · simp [stage2Loaded,active,Scratch.scalars]
  rw [values]
  have selected := pick_key (fun x : MidWord (Formats.format kind) =>
      (Formats.decodeStored kind x : WithTop Int)) (stage2Loaded (integerScratch kind p).scalars c mid) c.secondTree
  have coeTree : ∀ (tree : Tree c.midBlock) (input : Fin c.midBlock → Int),
      tree.pick orderedGt orderedEq (fun i => (input i : WithTop Int))=tree.pick orderedGt orderedEq input := by
    intro tree input
    induction tree with
    | leaf => rfl
    | fork a b ha hb => simp only [Tree.pick,ha,hb,leftWins,orderedGt,orderedEq,WithTop.coe_lt_coe,WithTop.coe_eq_coe]; rfl
  unfold stage2Selected
  exact congrArg Fin.val (selected.trans (coeTree c.secondTree
    (fun lane => Formats.decodeStored kind (stage2Loaded (integerScratch kind p).scalars c mid lane)))).symm

theorem stage1_reachable (kind : RealScratch.FloatingKind) (cast : Word → MidWord kind.format)
    (c : GlobalConfig) (input : Nat → Word) (pid : Nat) :
    Reachable kind cast (stage1Value (floatingScratch kind cast).scalars c input pid) :=
  Or.inl ⟨_,rfl⟩

/-- Integer store boundary is exactly the selected repair's source macro,
including NaN0/clamp/truncate and the requested-width scalar store. -/
theorem integer_cast (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (zero : ConversionMacro.ZeroClamp kind p) (word : Word) :
    ((ConversionMacro.SaturatingConversion.mk kind).emittedStore p word).cast (Formats.format_width kind)=
      (integerScratch kind p).cast word := by
  rw [ConversionMacro.emitted_source _ _ zero]
  rfl

#axiomsClean floating_selected
#axiomsClean integer_selected
#axiomsClean integer_cast
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedScalarBinding
