import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
noncomputable section

def RealSpec (r : Admission.Request) (bt : 0<r.shape.BT)
    (rt : CheckedHost.Runtime r) (realBefore : Nat → MemCell) (realScratch : BlockState)
    (scale : ℝ) (result : CheckedHost.Returned) : Prop :=
  ∃ a : CheckedHost.Allocation r,rt.allocate=.ok a ∧ result.objectId=a.layout.objectId ∧ result.tensor=a.tensor ∧
    Nonempty (Kernel.GridLaunchedOrdinary (DirectReal.kernel r.shape scale) (grid r.shape)
      (PhysicalDirect.realInput a.layout realBefore realScratch)
      (DirectReal.result r.shape scale (PhysicalDirect.realInput a.layout realBefore realScratch))) ∧
    (∀ h t v,PhysicalDirect.readReal (PhysicalDirect.realResult r.shape a.layout scale realBefore realScratch
      (a.layout.base "o"+(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val)))=
      Host.formula r.shape bt (Real.inputs regions (PhysicalDirect.realInput a.layout realBefore realScratch)) scale h t v) ∧
    (∀ p,¬(PhysicalDirect.output a.layout).Contains p →
      PhysicalDirect.realResult r.shape a.layout scale realBefore realScratch p=realBefore p) ∧
    (∀ region∈(["q","k","v","h","g"] : List RegionName),∀ i,i<capacity r.shape region →
      PhysicalDirect.realResult r.shape a.layout scale realBefore realScratch (a.layout.base region+i)=
        realBefore (a.layout.base region+i))

/-- Every successful wrapper return is the same allocator object whose physical
Real semantics executes the full repaired grid, satisfies the independent causal
contraction at every returned element, and preserves all caller input storage. -/
theorem returned_real (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (rt : CheckedHost.Runtime r) (before : Nat → BitVec 32) (fpScratch : State NativeGraph.Expr)
    (caller : Nat) (result : CheckedHost.Returned)
    (success : CheckedHost.run r rt before fpScratch caller=.ok result)
    (realBefore : Nat → MemCell) (realScratch : BlockState) (scale : ℝ) :
    RealSpec r (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1
      rt realBefore realScratch scale result := by
  have post:=(CheckedHost.wrapper_complete r rt before fpScratch caller).2
  rw [success] at post
  rcases post with ⟨a,allocation,identity,tensor,rest⟩
  have value:=PhysicalDirect.real_complete r.shape
    (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1 a.layout scale realBefore realScratch
  exact ⟨a,allocation,identity,tensor,value⟩

/-- Complete checked host result and the independent full returned Real tensor.
Real input/scalar interpretation is supplied separately from FP32 bit payloads. -/
theorem wrapper_correct (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (rt : CheckedHost.Runtime r) (before : Nat → BitVec 32) (fpScratch : State NativeGraph.Expr)
    (caller : Nat) (realBefore : Nat → MemCell) (realScratch : BlockState) (scale : ℝ) :
    CheckedHost.ResultSpec r rt before fpScratch caller (CheckedHost.run r rt before fpScratch caller) ∧
    (∀ result,CheckedHost.run r rt before fpScratch caller=.ok result →
      RealSpec r (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1
        rt realBefore realScratch scale result) :=
  ⟨(CheckedHost.wrapper_complete r rt before fpScratch caller).2,
    fun result success=>returned_real r checked rt before fpScratch caller result success realBefore realScratch scale⟩

abbrev checked_host := @CheckedHost.wrapper_complete
abbrev direct_source := @PhysicalDirect.real_complete
#axiomsClean wrapper_correct
#axiomsClean returned_real
#axiomsClean direct_source
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.Correct
