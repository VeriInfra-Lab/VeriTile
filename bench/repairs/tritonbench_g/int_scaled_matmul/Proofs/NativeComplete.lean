import bench.repairs.tritonbench_g.int_scaled_matmul.NativeIO
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.IOComplete
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.RawCompatibility

namespace VeriTile.Bench.Repairs.IntScaledMatmul.NativeComplete
open VeriTile.Triton FP.Structural Revision001 FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem int_equivalent (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) :
    (IntIO.io rest (blockParameters x)).Equivalent (NativeIO.intVariant rest x) := by
  have correct := IntIO.complete rest (blockParameters x) geometry
    (Contract.positive_tiles x.config legal).2.2.1 ha hb
  apply ProgramIO.EquivalentOn.of_implementsOn correct
  unfold NativeIO.intVariant ProgramIO.Variant.toProgramIO
  rw [NativeIO.intInvocations_eq rest x legal]
  exact correct

theorem scaled_equivalent (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c) :
    (ScaledIO.io rest undefined (scaledParameters x scale)).Equivalent
      (NativeIO.scaledVariant rest undefined x scale) := by
  have correct := ScaledIO.complete rest undefined (scaledParameters x scale) geometry
    (FiniteGrid.even_exact x scale legal)
    (LogicalSum.ceil_cover _ _ (Contract.positive_tiles x.config legal).2.2.1) ha hb hs
  apply ProgramIO.EquivalentOn.of_implementsOn correct
  unfold NativeIO.scaledVariant ProgramIO.Variant.toProgramIO
  rw [NativeIO.scaledInvocations_eq rest undefined x scale legal]
  exact correct

/-- Exact complete-memory transport to the original finite signed-index K32
instruction executor, including the original empty-grid shortcut. -/
theorem int_transport (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (initial : State Word) :
    ∃ original lowered,
      runSchedule ((List.range (Geometry.count (BlockWord.geometry (blockParameters x)))).map
        (NativeIO.intInvocation rest x)) initial=some original ∧
      RawFP.intNative x initial=some lowered ∧ original.mem=lowered.mem ∧
      GridExecution.Post x.c (Geometry.address (BlockWord.geometry (blockParameters x)))
        (fun i : BlockWord.LogicalIndex (blockParameters x) =>
          Cell.mk .int (BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val))
        (fun _=>True) initial original := by
  obtain ⟨original,lowered,executed,native,equal,post⟩ :=
    RawFP.int_matmul_equiv rest x legal envelope geometry ha hb initial
  refine ⟨original,lowered,?_,native,equal,post⟩
  simp only [NativeIO.intInvocations_eq rest x legal]
  exact (IntIO.schedule_equal rest (blockParameters x) _ initial).trans executed

theorem scaled_transport (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c) (initial : State Word) :
    ∃ original lowered,
      runSchedule ((List.range (Geometry.count (ScaledWord.geometry (scaledParameters x scale)))).map
        (NativeIO.scaledInvocation rest undefined x scale)) initial=some original ∧
      RawFP.scaledNative undefined x scale initial=some lowered ∧ original.mem=lowered.mem ∧
      GridExecution.Post x.c (Geometry.address (ScaledWord.geometry (scaledParameters x scale)))
        (fun i : ScaledWord.LogicalIndex (scaledParameters x scale) => Cell.mk .int
          (TotalScaled.logicalResult Native.sourceConfig (KernelModel.fallback undefined)
            (scaledParameters x scale) initial i.1.val i.2.val))
        (fun _=>True) initial original := by
  obtain ⟨original,lowered,executed,native,equal,post⟩ :=
    RawFP.int_scaled_matmul_equiv rest undefined scale x legal envelope geometry ha hb hs initial
  refine ⟨original,lowered,?_,native,equal,post⟩
  simp only [NativeIO.scaledInvocations_eq rest undefined x scale legal]
  exact (ScaledIO.schedule_equal rest undefined (scaledParameters x scale) _ initial).trans executed

#axiomsClean int_equivalent
#axiomsClean scaled_equivalent
#axiomsClean int_transport
#axiomsClean scaled_transport
end
end VeriTile.Bench.Repairs.IntScaledMatmul.NativeComplete
