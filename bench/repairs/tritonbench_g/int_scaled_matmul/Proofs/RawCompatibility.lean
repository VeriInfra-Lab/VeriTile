import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelModel
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelNative

/-! Previous complete raw statements, retained as ordinary compatibility
theorems. Their unchanged execution foundations also supply the shared IO. -/

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RawCorrect
open VeriTile.Triton FP.Structural
open Revision001 FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Unscaled signed32 whole-K mathematics, successful complete shared source,
and full memory frame over raw shape/stride geometry and region memory. -/
theorem int_matmul_correct (rest : Algebra Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State Word) :
    ∃ final, BlockWord.grid rest p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (BlockWord.geometry p))
        (fun i : BlockWord.LogicalIndex p =>
          Cell.mk .int (BlockWord.logicalDot p initial i.1.val i.2.val))
        (fun _=>True) initial final :=
  BlockWord.grid_correct rest p geometry positive ha hb initial

/-- Scaled signed32 whole-K mathematics with the exact retained FP32 promotion,
multiply and total output conversion. No overflow, tail or empty-grid inputs
are excluded and no Tensor objects occur in the raw interface. -/
theorem int_scaled_matmul_correct (rest : Algebra Word) (undefined : Word → BitVec 32)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (initial : State Word) :
    ∃ final, ScaledWord.grid rest Native.sourceConfig (KernelModel.fallback undefined) p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (ScaledWord.geometry p))
        (fun i : ScaledWord.LogicalIndex p => Cell.mk .int
          (TotalScaled.logicalResult Native.sourceConfig (KernelModel.fallback undefined) p initial i.1.val i.2.val))
        (fun _=>True) initial final :=
  TotalScaled.grid_correct rest Native.sourceConfig (KernelModel.fallback undefined) p
    geometry even cover ha hb hs initial

#axiomsClean int_matmul_correct
#axiomsClean int_scaled_matmul_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RawCorrect


namespace VeriTile.Bench.Repairs.IntScaledMatmul.RawFP
open VeriTile.Triton FP.Structural
open Revision001 FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Raw empty-grid handling is independent of Tensor admission. -/
def intNative (x : Launch) (s : State Word) : Option (State Word) :=
  if x.M=0 ∨ x.N=0 then some s else IntegerInstructions.intGrid x s

def scaledNative (undefined : Word → BitVec 32) (x : Launch) (scale : RegionName)
    (s : State Word) : Option (State Word) :=
  if x.M=0 ∨ x.N=0 then some s else CheckedNative.grid (KernelNative.ports undefined) x scale s

/-- Complete output values and the exterior frame determine the whole memory. -/
theorem post_unique {I : Type} (output : RegionName) (address : I → Nat)
    (value : I → Cell Word) (initial left right : State Word)
    (hl : GridExecution.Post output address value (fun _=>True) initial left)
    (hr : GridExecution.Post output address value (fun _=>True) initial right) : left.mem=right.mem := by
  classical
  funext region offset
  by_cases same : region=output
  · subst region
    by_cases hit : ∃ i,offset=address i
    · obtain ⟨i,rfl⟩ := hit
      exact (hl.1 i trivial).trans (hr.1 i trivial).symm
    · have outside : output≠output ∨ ∀ i,True → offset≠address i :=
        Or.inr (fun i _ equal => hit ⟨i,equal⟩)
      exact (hl.2 _ _ outside).trans (hr.2 _ _ outside).symm
  · exact (hl.2 _ _ (Or.inl same)).trans (hr.2 _ _ (Or.inl same)).symm

/-- Complete raw source/native relation with independent whole-K values and
memory frame. Signed-index bounds apply only to nonempty grids. -/
theorem int_matmul_equiv (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (initial : State Word) :
    ∃ original lowered,BlockWord.grid rest (blockParameters x) initial=some original ∧
      intNative x initial=some lowered ∧ original.mem=lowered.mem ∧
      GridExecution.Post x.c (Geometry.address (BlockWord.geometry (blockParameters x)))
        (fun i : BlockWord.LogicalIndex (blockParameters x) =>
          Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
        (fun _=>True) initial original := by
  obtain ⟨original,executed,post⟩ := BlockWord.grid_correct rest (blockParameters x) geometry
    (Contract.positive_tiles x.config legal).2.2.1 ha hb initial
  have native : ∃ lowered,intNative x initial=some lowered ∧
      GridExecution.Post x.c (Geometry.address (BlockWord.geometry (blockParameters x)))
        (fun i : BlockWord.LogicalIndex (blockParameters x) =>
          Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
        (fun _=>True) initial lowered := by
    by_cases empty : x.M=0 ∨ x.N=0
    · refine ⟨initial,by simp [intNative,empty],?_,?_⟩
      · intro i _
        have hm : i.1.val<x.M := i.1.isLt
        have hn : i.2.val<x.N := i.2.isLt
        rcases empty with h | h <;> omega
      · intro r o _;rfl
    · have env : Contract.IndexEnvelope x.M x.N x.K x.config := by
        rcases envelope with hm | hn | env
        · exact False.elim (empty (Or.inl hm))
        · exact False.elim (empty (Or.inr hn))
        · exact env
      obtain ⟨lowered,run,values⟩ := FiniteGrid.int_correct rest x legal env geometry ha hb initial
      refine ⟨lowered,?_,values⟩
      simpa only [intNative,if_neg empty,IntegerInstructions.intGrid_eq x legal] using run
  obtain ⟨lowered,loweredRun,loweredPost⟩ := native
  exact ⟨original,lowered,executed,loweredRun,post_unique _ _ _ initial original lowered post loweredPost,post⟩

/-- Complete raw source/native relation with independent whole-K values and
memory frame. Signed-index bounds apply only to nonempty grids. -/
theorem int_scaled_matmul_equiv (rest : Algebra Word) (undefined : Word → BitVec 32) (scale : RegionName) (x : Launch)
    (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c) (initial : State Word) :
    ∃ original lowered,ScaledWord.grid rest Native.sourceConfig (KernelModel.fallback undefined) (scaledParameters x scale) initial=some original ∧
      scaledNative undefined x scale initial=some lowered ∧ original.mem=lowered.mem ∧
      GridExecution.Post x.c (Geometry.address (ScaledWord.geometry (scaledParameters x scale)))
        (fun i : ScaledWord.LogicalIndex (scaledParameters x scale) =>
          Cell.mk .int (TotalScaled.logicalResult Native.sourceConfig (KernelModel.fallback undefined) (scaledParameters x scale) initial i.1.val i.2.val))
        (fun _=>True) initial original := by
  obtain ⟨original,executed,post⟩ := TotalScaled.grid_correct rest Native.sourceConfig (KernelModel.fallback undefined) (scaledParameters x scale) geometry
    (FiniteGrid.even_exact x scale legal)
    (LogicalSum.ceil_cover _ _ (Contract.positive_tiles x.config legal).2.2.1) ha hb hs initial
  have native : ∃ lowered,scaledNative undefined x scale initial=some lowered ∧
      GridExecution.Post x.c (Geometry.address (ScaledWord.geometry (scaledParameters x scale)))
        (fun i : ScaledWord.LogicalIndex (scaledParameters x scale) =>
          Cell.mk .int (TotalScaled.logicalResult Native.sourceConfig (KernelModel.fallback undefined) (scaledParameters x scale) initial i.1.val i.2.val))
        (fun _=>True) initial lowered := by
    by_cases empty : x.M=0 ∨ x.N=0
    · refine ⟨initial,by simp [scaledNative,empty],?_,?_⟩
      · intro i _
        have hm : i.1.val<x.M := i.1.isLt
        have hn : i.2.val<x.N := i.2.isLt
        rcases empty with h | h <;> omega
      · intro r o _;rfl
    · have env : Contract.IndexEnvelope x.M x.N x.K x.config := by
        rcases envelope with hm | hn | env
        · exact False.elim (empty (Or.inl hm))
        · exact False.elim (empty (Or.inr hn))
        · exact env
      obtain ⟨lowered,run,values⟩ := FiniteGrid.scaled_correct rest Native.sourceConfig (KernelModel.fallback undefined)
        x scale legal env geometry ha hb hs initial
      refine ⟨lowered,?_,values⟩
      simp only [scaledNative,if_neg empty,KernelNative.ports_eq,
        KernelExecution.native_scaled_grid,IntegerInstructions.scaledGrid_eq _ _ x scale legal]
      exact run
  obtain ⟨lowered,loweredRun,loweredPost⟩ := native
  exact ⟨original,lowered,executed,loweredRun,post_unique _ _ _ initial original lowered post loweredPost,post⟩

export KernelNative (multiply_comm instructions_eq)
#axiomsClean int_matmul_equiv
#axiomsClean int_scaled_matmul_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RawFP
