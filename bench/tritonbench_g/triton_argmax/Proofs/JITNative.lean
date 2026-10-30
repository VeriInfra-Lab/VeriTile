import bench.tritonbench_g.triton_argmax.Proofs.JITInteger
import bench.tritonbench_g.triton_argmax.Proofs.TypedScalarBinding
import bench.tritonbench_g.triton_argmax.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section

theorem first_index (s : State Word) (X : RegionName) (c : GlobalConfig)
    (g : Native.GlobalPrograms c) (base : Nat) (positive : 0<c.block) :
    Typed.firstIndex s X c.size c.block base positive=
      Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base) := by
  unfold Typed.firstIndex Native.firstIndex
  rw [TypedReduction.selected_eq positive _ _ (g.first _).numericComplete,Native.Sites.pick_eq]
  rfl

/-- Every shared first-stage quotation executes the same reduction and original
value reread as the live native program. The only format bridge is the local
scalar codec at the requested-width store. -/
theorem first_float (kind : TypedScratch.Kind) (round : TypedScratch.Rounding)
    (cast : Word → MidWord kind.format) (codec : TypedScalarBinding.FloatCodec kind round cast)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (base : Nat) (index64 : Bool) (positive : 0<c.block) :
    ∃ t, FP.Structural.exec (TypedScratch.backend round rest)
      (JITSource.stage1 X V I (JITStage1.floatingType kind) c.block index64) (JITStage1.arguments s c.size base)=some t ∧
      t.mem=((s.write V (s.pids 0+base) (.mk (TypedScratch.channel kind).toTileDType
        (ofFloat (TypedScratch.channel kind) (codec.decode
          (Native.firstValue (floatingScratch kind cast) g (fun i => (s.mem X i).read .real) (s.pids 0+base)))))).write
        (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat
          (Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base))))).mem := by
  obtain ⟨t,run,mem⟩ := JITStage1.floating_run kind round rest s X V I c.size c.block base index64 positive
  refine ⟨t,run,mem.trans ?_⟩
  simp only [TypedScratch.firstWritten,first_index s X c g base positive,
    Native.firstValue,floatingScratch,codec.castValue]

/-- Actual current scratch loads and the selected-index gather connect the
complete shared second quotation to the live native second program. -/
theorem second_float (kind : TypedScratch.Kind) (round : TypedScratch.Rounding)
    (cast : Word → MidWord kind.format) (codec : TypedScalarBinding.FloatCodec kind round cast)
    (rest : Algebra Word) (s : State Word) (V : RegionName) (I O : Region .int)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.midBlock)
    (mid : Nat → MidWord kind.format)
    (reachable : ∀ i,i<c.blocks → TypedScalarBinding.Reachable kind cast (mid i))
    (reads : ∀ i,i<c.blocks → toFloat (TypedScratch.channel kind)
      ((s.mem V i).read (TypedScratch.channel kind).toTileDType)=codec.decode (mid i)) :
    ∃ t, FP.Structural.exec (TypedScratch.backend round rest)
      (JITSource.stage2 V I O (JITStage1.floatingType kind) c.midBlock) (s.setReg "mid_size" .int [] (fun _ => (c.blocks : Int)))=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I)
        (Native.secondSelected (floatingScratch kind cast) Bool.true g mid).val).read .int))).mem := by
  obtain ⟨t,run,mem⟩ := JITStage1.floating_second kind round rest s V I O c.blocks c.midBlock positive
  refine ⟨t,run,mem.trans ?_⟩
  unfold TypedScratch.secondWritten
  rw [TypedScalarBinding.floating_selected kind round cast codec c positive
    (g.secondNumeric ▸ g.second.numericComplete) s V mid reachable reads,Native.second_selected]

theorem integer_selected (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (c : GlobalConfig) (positive : 0<c.midBlock) (complete : c.secondTree.Complete)
    (s : State Word) (V : RegionName) (mid : Nat → MidWord (Formats.format kind))
    (reads : ∀ i,i<c.blocks →
      JITInteger.read kind (s.mem V i)=Formats.decodeStored kind (mid i)) :
    JITInteger.secondIndex kind s V c.blocks c.midBlock=
      (stage2Selected (integerScratch kind p).scalars c mid).val := by
  have padding : Formats.decodeStored kind (integerScratch kind p).padding = kind.padding := by
    change Formats.decodeStored kind (Formats.paddingWord kind) = kind.padding
    cases kind <;> decide
  unfold JITInteger.secondIndex
  rw [IntegerReduction.shared_tree positive _ c.secondTree complete]
  have values : (fun lane : Fin c.midBlock => if lane.val<c.blocks then
      JITInteger.read kind (s.mem V lane.val) else kind.padding)=
      fun lane => Formats.decodeStored kind (stage2Loaded (integerScratch kind p).scalars c mid lane) := by
    funext lane
    by_cases active : lane.val<c.blocks
    · simp [stage2Loaded,active,reads lane.val active,Scratch.scalars]
    · simp [stage2Loaded,active,Scratch.scalars,padding]
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


theorem second_integer (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (V : RegionName) (I O : Region .int)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.midBlock)
    (mid : Nat → MidWord (Formats.format kind))
    (reads : ∀ i,i<c.blocks → JITInteger.read kind (s.mem V i)=Formats.decodeStored kind (mid i)) :
    ∃ t, FP.Structural.exec rest (JITSource.stage2 V I O (JITInteger.sourceType kind) c.midBlock)
      (s.setReg "mid_size" .int [] (fun _ => (c.blocks : Int)))=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I)
        (Native.secondSelected (integerScratch kind p) Bool.false g mid).val).read .int))).mem := by
  obtain ⟨t,run,mem⟩ := JITInteger.second_run kind rest s V I O c.blocks c.midBlock
  refine ⟨t,run,mem.trans ?_⟩
  rw [integer_selected kind p c positive (g.secondNumeric ▸ g.second.numericComplete)
    s V mid reads,Native.second_selected]

#axiomsClean first_float
#axiomsClean second_float
#axiomsClean second_integer
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNative
