import bench.tritonbench_g.chunk_retention_ops.Proofs.IOSafety
import bench.tritonbench_g.chunk_retention_ops.Proofs.IOExecution
import bench.tritonbench_g.chunk_retention_ops.Proofs.IORealFormula
import bench.tritonbench_g.chunk_retention_ops.Proofs.IONativeFormula
import bench.tritonbench_g.chunk_retention_ops.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelPacket
open KernelNative
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 30000

private theorem mapValue_identity {α : Type} (d : TileDType) (x : Value α d) :
    mapValue id d x=x := by cases d <;> rfl

/-- Covering the complete old footprint gives the exact typed-cell exterior
frame of the finite public IO. -/
theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (a : Call) (m : KernelPacket.Memory α) (r : RegionName) (off : Nat)
    (outside : ∀ port j, outputActive a port j →
      r ≠ (outputs a.shape)[port].buf ∨ off ≠ outputOffset a port j) :
    KernelPacket.reference M D a m r off=m r off := by
  apply KernelPacket.frame
  intro written
  obtain ⟨port,j,active,region,offset⟩:=footprint_covered a r off written
  exact (outside port j active).elim (fun h=>h region) (fun h=>h offset)

theorem real_result (a : Call)
    (chunk : a.stage=.backwardGradient → a.pid.y<a.shape.chunks)
    (xs : Inputs ℝ a.shape) (s : State ℝ) (binding : (realIO a).InputBinding xs s) :
    ∃ final,runSchedule ((realIO a).schedule xs) s=some final ∧
      (realIO a).OutputBinding xs (mathematical a xs) final ∧ (realIO a).Frame xs s final := by
  obtain ⟨final,ran,result⟩:=schedule_result KernelReal.algebra KernelReal.dotPrimitive a s
  refine ⟨final,ran,?_,?_⟩
  · intro port j active
    change mapCell id (final.mem (outputs a.shape)[port].buf (outputOffset a port j))=
      Cell.mk (outputs a.shape)[port].dtype (mathematical a xs port j)
    rw [result]
    apply Eq.trans (congrArg (mapCell id)
      (reference_read KernelReal.algebra KernelReal.dotPrimitive a s.mem port j active))
    change Cell.mk _ (mapValue id _ _)=_
    rw [mapValue_identity]
    apply congrArg (Cell.mk _)
    exact (real_scalar_values a s.mem port j active).trans
      ((scalar_transport realOps a.scale a xs _ (real_reads a xs s binding) port j active).trans
        (scalar_mathematical a xs chunk port j active))
  · intro r off outside
    rw [result]
    exact reference_frame KernelReal.algebra KernelReal.dotPrimitive a s.mem r off outside.1

/-- The same original source run supplies termination, safety, the independent
mathematical outputs and exact frame. Only the original gradient chunk premise
is needed by the dense algebraic formula. -/
theorem real_implements (a : Call)
    (chunk : a.stage=.backwardGradient → a.pid.y<a.shape.chunks) :
    (realIO a).Implements (mathematical a) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded s _
    exact schedule_safe KernelReal.algebra KernelReal.dotPrimitive id id a xs bounds bounded s
  · exact real_result a chunk

theorem native_result (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call)
    (xs : Inputs Word a.shape) (s : State Term) (binding : (nativeIO isa literal a).InputBinding xs s) :
    ∃ final,runSchedule ((nativeIO isa literal a).schedule xs) s=some final ∧
      (nativeIO isa literal a).OutputBinding xs (nativeFormula isa literal a xs) final ∧
      (nativeIO isa literal a).Frame xs s final := by
  obtain ⟨final,ran,result⟩:=schedule_result model (dotModel a) a s
  let m : RegionName→Nat→Word:=fun r off=>S.lower (sharedOps (wordOps isa)) literal ((s.mem r off).read .real)
  have reads:=native_readEq isa literal a xs s binding
  refine ⟨final,ran,?_,?_⟩
  · intro port j active
    change mapCell (S.lower (sharedOps (wordOps isa)) literal)
      (final.mem (outputs a.shape)[port].buf (outputOffset a port j))=
      Cell.mk (outputs a.shape)[port].dtype (nativeFormula isa literal a xs port j)
    rw [result]
    apply Eq.trans (congrArg (mapCell (S.lower (sharedOps (wordOps isa)) literal))
      (reference_read model (dotModel a) a s.mem port j active))
    change Cell.mk _ (mapValue (S.lower (sharedOps (wordOps isa)) literal) _ _)=_
    apply congrArg (Cell.mk _)
    apply Eq.trans (congrArg (mapValue (S.lower (sharedOps (wordOps isa)) literal) _)
      (source_values_readEq model (dotModel a) a s.mem (input m) reads port j active))
    exact (native_scalar_values isa literal laws a m port j active).trans
      (scalar_transport (wordOps isa) (literal a.scale) a xs m
        (native_reads isa literal a xs s binding) port j active)
  · intro r off outside
    rw [result]
    exact reference_frame model (dotModel a) a s.mem r off outside.1

/-- Native observation retains every arbitrary program coordinate admitted by
KernelNative.correct, the same FP32 scalar/literal boundaries, and all eight
finite output families. Access safety is proved from the declared IO bounds. -/
theorem native_implements (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call) :
    (nativeIO isa literal a).Implements (nativeFormula isa literal a) := by
  apply AddressedProgramIO.Implements.intro
  · intro xs bounds bounded s _
    exact schedule_safe model (dotModel a) (fun _ : Word=>default)
      (S.lower (sharedOps (wordOps isa)) literal) a xs bounds bounded s
  · exact native_result isa literal laws a

theorem native_boundary (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : S.LiteralLaws (sharedOps (wordOps (Numeric.referenceISA u))) literal) (a : Call) :
    (nativeIO (Numeric.nativeISA ports) literal a).Implements
      (nativeFormula (Numeric.referenceISA u) literal a) := by
  rw [Numeric.isa_eq u ports laws]
  exact native_implements (Numeric.referenceISA u) literal constants a

#axiomsClean real_implements
#axiomsClean native_implements
#axiomsClean native_boundary
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
