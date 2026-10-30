import bench.repairs.tritonbench_g.chunk_retention.Proofs.IOSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.IOResults

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]

theorem real_implements (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid) :
    (realIO a p grid).Implements (realFormula a p) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded s _
    let start := (launch (engine (withDot KernelReal.algebra KernelReal.dot)) a p grid).initial s
    have hs := source_safe KernelReal.algebra KernelReal.dot id id a start legal xs bounds bounded
    simpa only [realIO,io,scheduleSafe_singleton,Invocation.Safe] using hs
  · exact fun xs s binding=>real_result a p grid legal xs s binding

theorem native_implements (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid) :
    (nativeIO ops literal a p grid).Implements (nativeFormula ops literal a p) := by
  apply AddressedProgramIO.Implements.intro
  · intro xs bounds bounded s _
    let start := (launch (engine (withDot (algebra ops.zero) (nativeDot a))) a p grid).initial s
    have hs := source_safe (algebra ops.zero) (nativeDot a) (Term.input "" 0) (lower ops literal)
      a start legal xs bounds bounded
    simpa only [nativeIO,scheduleSafe_singleton,Invocation.Safe] using hs
  · exact fun xs s binding=>native_result ops literal laws castZero a p grid legal xs s binding

theorem fp_implements (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32)
    (literals : LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid) :
    (nativeIO (wordOps (Numeric.nativeISA ports)) literal a p grid).Implements
      (nativeFormula (wordOps (Numeric.referenceISA u)) literal a p) := by
  have result := native_implements (wordOps (Numeric.nativeISA ports)) literal literals rfl a p grid legal
  simpa only [Numeric.isa_eq u ports laws] using result

#axiomsClean real_implements
#axiomsClean native_implements
#axiomsClean fp_implements
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
