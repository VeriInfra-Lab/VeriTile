import bench.tritonbench_g.sgmv_expand_slice.Candidate
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def abi (add : Bool) : SourceABI :=
  ⟨.single,.half,.single,.w32,.w32,.w32,add⟩

theorem original_carrier (a : SourceABI) (hi : a.input=.single)
    (hw : a.weight=.half) (ho : a.output=.single) (p : KernelArgs) :
    VeriTile.Bench.Repairs.SgmvExpandSlice.kernel a p =
      VeriTile.Bench.Repairs.SgmvExpandSlice.kernel (abi a.addInputs) p := by
  rcases a with ⟨input,weight,out,sw,lw,aw,add⟩
  dsimp only at hi hw ho
  subst input; subst weight; subst out
  cases sw <;> cases lw <;> cases aw <;> rfl

theorem candidate_carrier (a : SourceABI) (p : KernelArgs) :
    IndexReuse.kernel a p = IndexReuse.kernel (abi a.addInputs) p := by
  rcases a with ⟨input,weight,out,sw,lw,aw,add⟩
  cases sw <;> cases lw <;> cases aw <;> rfl

def active (add : Bool) (p : KernelArgs) :=
  SourceParts.thenBody (SourceParts.thenBody
    (SourceParts.thenBody (IndexReuse.kernel (abi add) p).surfaceBody 16) 0) 1

def suffix (add : Bool) (p : KernelArgs) := (active add p).drop 8

/-- These are the actual first three suffix assignments. Reusing the existing
integer vector offsets produces precisely the same successor state. -/
theorem indices_eq (M : Algebra α) (add : Bool) (p : KernelArgs)
    (c : SourceSuffix.Coordinates) (s t : State α) (acc : Values α .real [32,32])
    (h : SourceSuffix.Ready p c s t acc)
    (hm : t.regs .int [32] "offset_m" = some (fun i => (SourceSetup.rowIndex c.tileRow i.1 : Int)))
    (hn : t.regs .int [32] "offset_n" = some (fun i => (SourceSetup.columnIndex c.tileColumn i.1 : Int))) :
    run M ((suffix add p).take 3) t = run M ((SourceParts.suffix (abi add) p).take 3) t := by
  simp [suffix,active,IndexReuse.kernel,IndexReuse.sourceFace,IndexReuse.sourceHalfSingle323232,
    SourceParts.suffix,SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,
    SourceParts.thenBody,SourceParts.outer,VeriTile.Bench.Repairs.SgmvExpandSlice.kernel,sourceFace,
    sourceHalfSingle323232,abi,ComputeKernel.surfaceBody,ScalarFormat.dtype,LowFormat.scalar,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,
    FP.Structural.bop,numericLt,Region.cast,ofFloat,toFloat,SourceSetup.rowIndex,SourceSetup.columnIndex,
    hm,hn,h.start,h.row,h.column,h.slice,h.acc,Int.natCast_add,Int.natCast_mul,Int.add_assoc]
  unfold FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,Int.add_assoc]

/-- The remaining loads, arithmetic, masks and stores are unchanged source. -/
theorem suffix_eq (M : Algebra α) (add : Bool) (p : KernelArgs)
    (c : SourceSuffix.Coordinates) (s t : State α) (acc : Values α .real [32,32])
    (h : SourceSuffix.Ready p c s t acc)
    (hm : t.regs .int [32] "offset_m" = some (fun i => (SourceSetup.rowIndex c.tileRow i.1 : Int)))
    (hn : t.regs .int [32] "offset_n" = some (fun i => (SourceSetup.columnIndex c.tileColumn i.1 : Int))) :
    run M (suffix add p) t = run M (SourceParts.suffix (abi add) p) t := by
  have sameTail : (suffix add p).drop 3 = (SourceParts.suffix (abi add) p).drop 3 := by rfl
  conv_lhs => rw [← List.take_append_drop 3 (suffix add p)]
  conv_rhs => rw [← List.take_append_drop 3 (SourceParts.suffix (abi add) p)]
  simp only [run_append,indices_eq M add p c s t acc h hm hn,sameTail]

theorem suffix_safe_iff (M : Algebra α) (add : Bool) (p : KernelArgs)
    (c : SourceSuffix.Coordinates) (s t : State α) (acc : Values α .real [32,32])
    (h : SourceSuffix.Ready p c s t acc)
    (hm : t.regs .int [32] "offset_m" = some (fun i => (SourceSetup.rowIndex c.tileRow i.1 : Int)))
    (hn : t.regs .int [32] "offset_n" = some (fun i => (SourceSetup.columnIndex c.tileColumn i.1 : Int)))
    (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (suffix add p) t ↔
      Safety.traceSafeList M bounds (SourceParts.suffix (abi add) p) t := by
  have sameTail : (suffix add p).drop 3 = (SourceParts.suffix (abi add) p).drop 3 := by rfl
  have prefixSafe : Safety.traceSafeList M bounds ((suffix add p).take 3) t ↔
      Safety.traceSafeList M bounds ((SourceParts.suffix (abi add) p).take 3) t := by
    simp [suffix,active,IndexReuse.kernel,IndexReuse.sourceFace,IndexReuse.sourceHalfSingle323232,
      SourceParts.suffix,SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,
      SourceParts.thenBody,SourceParts.outer,VeriTile.Bench.Repairs.SgmvExpandSlice.kernel,sourceFace,
      sourceHalfSingle323232,abi,ComputeKernel.surfaceBody,ScalarFormat.dtype,LowFormat.scalar,
      Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold]
  conv_lhs => rw [← List.take_append_drop 3 (suffix add p)]
  conv_rhs => rw [← List.take_append_drop 3 (SourceParts.suffix (abi add) p)]
  simp only [Safety.traceSafeList_append,prefixSafe,indices_eq M add p c s t acc h hm hn,sameTail]

#axiomsClean indices_eq
#axiomsClean suffix_eq
#axiomsClean suffix_safe_iff
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
