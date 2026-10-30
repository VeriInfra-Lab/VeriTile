import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedIntegerMacro
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The shared first-stage quotation with the integer store port. The source's
whole NaN0/clamp/truncate/width-store macro is interpreted at the one scalar
floatToInt port below. This is a named macro projection, not an assertion that
a literal unchecked float-to-int instruction implements the safe conversion. -/
def kernel := TypedSource.stage1Integer

def backend (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) : Algebra Word :=
  { Typed.backend rest with
    floatToInt := fun _ _ _ => some (fun word => some ((ConversionMacro.SaturatingConversion.mk kind).native p word)) }

def written (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := Typed.firstIndex s X size block base positive
  (s.write V (s.pids 0+base) (.mk .int
    ((ConversionMacro.SaturatingConversion.mk kind).native p ((s.mem X selected).read .real)))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

theorem source_run (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend kind p rest) (kernel X V I size block base index64) s=some t ∧
      t.mem=(written kind p s X V I size block base positive).mem := by
  cases index64 <;>
    simp [kernel,TypedSource.stage1Integer,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,Typed.backend,State.setReg,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,Typed.globalLoaded,Typed.firstIndex,written,Option.bind,
      store,TileShape.allIndices,Region.cast,collectValues?,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast, State.write, State.setReg_mem]
  all_goals rfl

theorem source_native (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (base : Nat) (index64 : Bool) (positive : 0<c.block) :
    ∃ t, FP.Structural.exec (backend kind p rest) (kernel X V I c.size c.block base index64) s=some t ∧
      t.mem=((s.write V (s.pids 0+base) (.mk .int (Formats.decodeStored kind
        (Native.firstValue (integerScratch kind p) g (fun i => (s.mem X i).read .real) (s.pids 0+base))))).write
        (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat
          (Native.firstIndex g (fun i => (s.mem X i).read .real) (s.pids 0+base))))).mem := by
  obtain ⟨t,run,mem⟩ := source_run kind p rest s X V I c.size c.block base index64 positive
  refine ⟨t,run,mem.trans ?_⟩
  rw [written,TypedNative.first_index s X c g base positive]
  rfl

#axiomsClean source_run
#axiomsClean source_native
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedIntegerMacro
