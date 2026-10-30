import bench.tritonbench_g.chunk_retention_ops.Sources.KernelPacket
import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeLocal
import bench.tritonbench_g.chunk_retention_ops.Proofs.Numeric
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open KernelPacket (Call)
noncomputable section
abbrev Word := BitVec 32
abbrev Term := S.Term Word
abbrev Memory := KernelPacket.Memory

def input (m : RegionName→Nat→Word) : Memory Term :=
  fun r o=>.mk .real (.input r o (m r o))

def model : Algebra Term := S.algebra (0 : Word)
/-- The selected transposed dK dot ends its live accumulator lifetime at its
local layout materialization. All other source dots use the ordered live FMA
profile. This is a primitive/compiler model boundary, not a result premise. -/
def dotModel (a : Call) : DotPrimitive Term := match a.stage with
  | .backwardGradient=>⟨(S.gradientDot (α:=Word)).apply⟩
  | _=>⟨(S.dot (α:=Word)).apply⟩

/-- Raw-word observation under the fixed scalar ISA interpretation. Literal
encoding covers source constants and the already selected launch scalar. -/
def observe (isa : ISA) (literal : ℝ→Word) (m : Memory Term) (r : RegionName) (o : Nat) : Word :=
  S.lower (sharedOps (wordOps isa)) literal ((m r o).read .real)

/-- An input-only native operation reference. It retains casts, source load
boundaries, per-key-block dV, and local FMA accumulation/materialization order.
It contains no source syntax, execution, compiler equality, or result oracle. -/
def target (isa : ISA) (literal : ℝ→Word) (a : Call) (m : RegionName→Nat→Word) : RegionName→Nat→Word :=
  observe isa literal (KernelPacket.reference model (dotModel a) a (input m))

theorem input_observe (isa : ISA) (literal : ℝ→Word) (m : RegionName→Nat→Word)
    (r : RegionName) (o : Nat) : observe isa literal (input m) r o=m r o := rfl

/-- All four selected quotations execute in the same staged scalar model used
by the local lowering profile. Bit equality requires only fixed scalar laws;
exact Real arithmetic is proved separately. -/
theorem correct (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word) (a : Call)
    (m : RegionName→Nat→Word) :
    ∃ t,FP.Structural.exec (engine (withDot model (dotModel a))) a.kernel (a.state (input m))=some t ∧
      (∀ r off,observe (Numeric.nativeISA ports) literal t.mem r off=
        target (Numeric.referenceISA u) literal a m r off) ∧
      (∀ r off,¬KernelPacket.Footprint a r off →
        observe (Numeric.nativeISA ports) literal t.mem r off=m r off) := by
  obtain ⟨t,ht,hm,hframe⟩:=KernelPacket.complete model (dotModel a) a (input m)
  refine ⟨t,ht,?_,?_⟩
  · intro r o
    rw [hm,Numeric.isa_eq u ports laws]
    rfl
  · intro r o safe
    unfold observe
    rw [hframe r o safe]
    rfl

/-- Satisfiability witness: the native primitive boundary accepts the reference
implementation without imposing any equality to exact Real arithmetic. -/
def referencePorts (u : Numeric.UnaryModels) : Numeric.ScalarPorts :=
  ⟨Numeric.integer64,Numeric.add32,Numeric.sub32,Numeric.mul32,Numeric.fma32,
    u.exp2FTZ,u.log2Device,u.hostScale⟩
theorem reference_laws (u : Numeric.UnaryModels) : Numeric.ScalarLaws u (referencePorts u) := by
  constructor <;> intros <;> rfl

#axiomsClean correct
#axiomsClean reference_laws
#specNonCircular target avoiding [run,FP.Structural.exec,KernelPacket.source,KernelPacket.complete,correct]
#specNonCircular Numeric.ScalarLaws avoiding [target,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
