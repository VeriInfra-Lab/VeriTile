import bench.repairs.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.RealIO
import bench.tritonbench_g.context_attn_fwd.Proofs.HostComposition
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.KernelRawGrid
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.KernelProfile
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeTransport

/-! Retained optional raw/flat consequences. These are the original statements,
using the same complete source execution and safety foundations as the public IO.
They are not additional public specifications. -/

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RawCompatibility
open VeriTile Triton
noncomputable section

/-- Complete source execution, independent initial-array base-2 weighted sum,
active output values and full flat-cell frame. Geometry, allocation bounds,
raw loaded arrays and active-address injectivity are the complete inputs. -/
theorem correct_real (p : Layout) (r : Regions) (scale : ℝ) (positive : 0<p.BN) :
    Spec.Real (VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.IO.Implements (VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.sourceIO p r .fp32 Bool.false scale)
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.closedExpected p scale)) :=
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.implements_closed p r .fp32 Bool.false scale positive

namespace RawGrid
open VeriTile.Bench.TritonBenchG.ContextAttnFwd

def active (p : Layout) (r : Regions) (g : Grid) (s : BlockState)
    (idx : GridIndex g) (i : RealIO.Lane p) : Prop :=
  RealIO.active p (Program.launch (s.withGridIndex idx)) (Program.inputs s r) i

def address (p : Layout) (r : Regions) (A : FlatAlloc) (g : Grid) (s : BlockState)
    (idx : GridIndex g) (i : RealIO.Lane p) : MemCellAddr :=
  (A.flat,A.addr r.out (RealIO.write p (Program.launch (s.withGridIndex idx)) (Program.inputs s r) i))

def expected (p : Layout) (r : Regions) (scale : ℝ) (g : Grid) (s : BlockState)
    (idx : GridIndex g) (i : RealIO.Lane p) : ℝ :=
  RealIO.closedExpected p scale (Program.launch (s.withGridIndex idx)) (Program.inputs s r) i

end RawGrid

/-- Complete raw finite-grid composition. The hypotheses are only allocation,
formed-address bounds and unique active output ownership. -/
theorem correct_real_grid (p : Layout) (r : Regions) (scale : ℝ) (positive : 0<p.BN)
    (A : FlatAlloc) (allocation : A.Disjoint) (covered : ∀ rg,rg∉A.regions → A.extent rg=0)
    (g : Grid) (before : BlockState)
    (bounds : ∀ idx : GridIndex g,
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.AddressBounds.Valid p r
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.launch (before.withGridIndex idx))
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs before r) A.extent)
    (unique : ∀ idx jdx i j,RawGrid.active p r g before idx i → RawGrid.active p r g before jdx j →
      RawGrid.address p r A g before idx i=RawGrid.address p r A g before jdx j → idx=jdx ∧ i=j) :
    Spec.Real (∃ final,Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (kernel p r .fp32 scale).eraseDType.toAlgKernel) g (A.flattenState before) final) ∧
      (∀ idx i,RawGrid.active p r g before idx i →
        final.readMem (RawGrid.address p r A g before idx i).1 (RawGrid.address p r A g before idx i).2=
          RawGrid.expected p r scale g before idx i) ∧
      BlockState.WriteWithin
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostComposition.allWrites g
          (RawGrid.active p r g before) (RawGrid.address p r A g before)) (A.flattenState before) final) := by
  let io := VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.sourceIO p r .fp32 Bool.false scale
  have per : ∀ idx : GridIndex g,∃ final,
      exec (A.flattenKernel (kernel p r .fp32 scale).eraseDType.toAlgKernel)
        ((A.flattenState before).withGridIndex idx)=some final ∧
      (∀ i,RawGrid.active p r g before idx i →
        final.readMem (RawGrid.address p r A g before idx i).1 (RawGrid.address p r A g before idx i).2=
          RawGrid.expected p r scale g before idx i) ∧
      BlockState.WriteWithin
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostComposition.writes g
          (RawGrid.active p r g before) (RawGrid.address p r A g before) idx)
        ((A.flattenState before).withGridIndex idx) final := by
    intro idx
    obtain ⟨final,ran,values,frame⟩ := correct_real p r scale positive A allocation covered
      (before.withGridIndex idx) (by
        intro i j hi hj equal
        exact (unique idx idx i j hi hj (Prod.ext rfl (congrArg (A.addr r.out) equal))).2)
      (bounds idx)
    refine ⟨final,ran,values,?_⟩
    intro rg off outside
    apply frame rg off
    rintro ⟨flat,reg,offset,⟨i,active,eqReg,eqOff⟩,eqAddr⟩
    apply outside
    change r.out=reg at eqReg
    subst reg
    change VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.write p
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.launch (before.withGridIndex idx))
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs before r) i=offset at eqOff
    subst offset
    exact ⟨i,active,Prod.ext flat.symm eqAddr⟩
  exact VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostComposition.assemble g _ _
    (RawGrid.active p r g before) (RawGrid.address p r A g before)
    (RawGrid.expected p r scale g before) per
    (fun idx jdx i j hi hj eq => (unique idx jdx i j hi hj eq).1)

#axiomsClean correct_real_grid
alias shared_source := actual_source
#axiomsClean correct_real
#axiomsClean shared_source
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RawCompatibility

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RawCompatibility
open VeriTile Triton FP.Structural NativeGraph
noncomputable section
open Classical

/-- Native output from the independent masked recurrence and the same fixed
SM90 graph plan. It depends only on initial raw memory and kernel parameters. -/
def nativeTarget (p : Layout) (r : Regions) (scale : ℝ) (before : State Expr)
    (ops : ScalarOps) (i : TileIndex [p.BM,p.D]) (owner : Owner) : Word :=
  evaluate ops (NativeSM90.plan p.BM p.D)
    (FPReference.output p (backend (NativeSM90.plan p.BM p.D))
      (dots.support (scalar (NativeSM90.plan p.BM p.D)) p)
      scale (FPReference.inputs p r before) i) owner

/-- Complete typed source, native output words at every active lane/consumer,
and full tagged-cell frame. Only raw active write injectivity is required. -/
theorem correct_fp32 (p : Layout) (r : Regions) (scale : ℝ) (positive : 0<p.BN)
    (before : State Expr) (ops : ScalarOps)
    (unique : ∀ i j,FPSuccess.active p r before i → FPSuccess.active p r before j →
      FPSuccess.offset p r before i=FPSuccess.offset p r before j → i=j) :
    ∃ final,exec ((backend (NativeSM90.plan p.BM p.D)).withDefaultPrecision .fp32)
        (kernel p r .fp32 scale) before=some final ∧
      (∀ i,FPSuccess.active p r before i → ∀ owner,
        evaluate ops (NativeSM90.plan p.BM p.D)
          ((final.mem r.out (FPSuccess.offset p r before i)).read .real) owner=
          nativeTarget p r scale before ops i owner) ∧
      (∀ rg off,(¬∃ i,FPSuccess.active p r before i ∧ rg=r.out ∧ off=FPSuccess.offset p r before i) →
        final.mem rg off=before.mem rg off) := by
  obtain ⟨final,ran,memory⟩ := NativeTransport.single p r scale
    (backend (NativeSM90.plan p.BM p.D))
    (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) positive before
  refine ⟨final,ran,?_,?_⟩
  · intro i active owner
    rw [memory]
    have value := VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_readback_active
      r.out (FPSuccess.offset p r before)
      (fun i => Cell.mk .real (FPReference.output p (backend (NativeSM90.plan p.BM p.D))
        (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) scale (FPReference.inputs p r before) i))
      (FPSuccess.active p r before) before i active unique
    change evaluate ops _ (((FPSuccess.scatter _ _ _ _ _).mem _ _).read .real) owner=_
    rw [show (FPSuccess.scatter p r before before _).mem r.out (FPSuccess.offset p r before i)=_ from value]
    rfl
  · intro rg off outside
    rw [memory]
    exact FPSuccess.scatter_frame p r before before _ rg off
      (fun i hi hr ho => outside ⟨i,hi,hr,ho⟩)

/-- Complete raw grid execution agrees on every cell with the independent
native recurrence and preserves every cell outside its exact write footprint. -/
theorem correct_fp32_grid (p : Layout) (r : Regions) (scale : ℝ) (positive : 0<p.BN)
    (g : Grid) (jobs : List (GridIndex g)) (before : State Expr) :
    ∃ final,KernelRawGrid.run p r scale (backend (NativeSM90.plan p.BM p.D)) g jobs before=some final ∧
      final.mem=(KernelRawGrid.reference p r scale (backend (NativeSM90.plan p.BM p.D))
        (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) g jobs before).mem ∧
      (∀ rg off,KernelRawGrid.Outside p r scale (backend (NativeSM90.plan p.BM p.D))
        (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) g jobs before rg off →
        final.mem rg off=before.mem rg off) :=
  NativeTransport.complete p r scale (backend (NativeSM90.plan p.BM p.D))
    (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) positive g jobs before

#axiomsClean correct_fp32_grid
#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,kernel,FPSuccess.runGrid]
#axiomsClean correct_fp32
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RawCompatibility
