import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedScalarBinding
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedNative
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
      (TypedScratch.firstKernel kind X V I c.size c.block base index64) s=some t ∧
      t.mem=((s.write V (s.pids 0+base) (.mk (TypedScratch.channel kind).toTileDType
        (ofFloat (TypedScratch.channel kind) (codec.decode
          (Native.firstValue (floatingScratch kind cast) g (fun i => (s.mem X i).read .real) (s.pids 0+base)))))).write
        (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat
          (Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base))))).mem := by
  obtain ⟨t,run,mem⟩ := TypedScratch.stage1_run kind round rest s X V I c.size c.block base index64 positive
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
      (TypedScratch.secondKernel kind V I O c.blocks c.midBlock) s=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I)
        (Native.secondSelected (floatingScratch kind cast) Bool.true g mid).val).read .int))).mem := by
  obtain ⟨t,run,mem⟩ := TypedScratch.stage2_run kind round rest s V I O c.blocks c.midBlock positive
  refine ⟨t,run,mem.trans ?_⟩
  unfold TypedScratch.secondWritten
  rw [TypedScalarBinding.floating_selected kind round cast codec c positive
    (g.secondNumeric ▸ g.second.numericComplete) s V mid reachable reads,Native.second_selected]

theorem second_integer (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (V I O : Region .int)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.midBlock)
    (mid : Nat → MidWord (Formats.format kind))
    (reads : ∀ i,i<c.blocks → (s.mem (Region.cast V) i).read .int=Formats.decodeStored kind (mid i)) :
    ∃ t, FP.Structural.exec rest (TypedInteger.stage2 V I O c.blocks c.midBlock
      (Formats.decodeStored kind (integerScratch kind p).padding)) s=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I)
        (Native.secondSelected (integerScratch kind p) Bool.false g mid).val).read .int))).mem := by
  obtain ⟨t,run,mem⟩ := TypedInteger.stage2_run rest s V I O c.blocks c.midBlock
    (Formats.decodeStored kind (integerScratch kind p).padding)
  refine ⟨t,run,mem.trans ?_⟩
  rw [TypedScalarBinding.integer_selected kind p c positive (g.secondNumeric ▸ g.second.numericComplete)
    s V mid reads,Native.second_selected]

/-- Integer/bool specialization executes the shared source through its
selected original scalar, then uses precisely the original requested scalar
NaN0/clamp/truncate/store macro. No conversion outcome is assumed. -/
theorem first_integer (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (zero : ConversionMacro.ZeroClamp kind p) (rest : Algebra Word) (s : State Word)
    (X V : RegionName) (I : Region .int) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (base : Nat) (index64 : Bool) (positive : 0<c.block) :
    ∃ t, run (Typed.backend rest) (TypedInteger.firstPrefix X V I c.size c.block base index64) s=some t ∧
      t.mem=s.mem ∧
      t.regs .real [] "max_val"=some (fun _ => (s.mem X
        (Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base))).read .real) ∧
      t.regs .nat [] "max_index"=some (fun _ => Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base)) ∧
      ((ConversionMacro.SaturatingConversion.mk kind).emittedStore p ((s.mem X
        (Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base))).read .real)).cast (Formats.format_width kind)=
        Native.firstValue (integerScratch kind p) g (fun i => (s.mem X i).read .real) (s.pids 0+base) := by
  obtain ⟨t,run,mem,value,index⟩ := TypedInteger.first_prefix rest s X V I c.size c.block base index64 positive
  rw [first_index s X c g base positive] at value index
  exact ⟨t,run,mem,value,index,TypedScalarBinding.integer_cast kind p zero _⟩

#axiomsClean first_float
#axiomsClean second_float
#axiomsClean first_integer
#axiomsClean second_integer
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedNative
