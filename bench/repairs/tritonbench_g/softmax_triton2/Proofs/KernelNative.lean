import bench.repairs.tritonbench_g.softmax_triton2.Kernels
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.KernelNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 NativeGraph
noncomputable section

/-- Internal adapter for the reused source proof; only raw address geometry varies. -/
def descriptor (region : RegionName) (rows cols offset rowStride colStride : Nat) : Tensor :=
  ⟨0,region,rows,cols,offset,rowStride,colStride,0,Bool.false⟩

/-- This task's shared source on raw region/stride geometry. -/
def evaluate (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) (s : State Word) : Option (State Word) :=
  (exec (Floating.backend (algebra inst 1))
    (SoftmaxTriton2.softmax_kernel out inp ib ob ir ic orow oc cols B)
    (Floating.launchState (descriptor inp rows cols ib ir ic) p (liftState s))).map
      (projectState 1 B (descriptor out rows cols ob orow oc) p s)

/-- Descriptor reuse is exactly the same source evaluator. -/
theorem evaluator_identity (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) (s : State Word) :
    evaluate inst out inp rows cols ib ob ir ic orow oc B p s =
      NativeGraph.evaluate inst 1 (descriptor inp rows cols ib ir ic)
        (descriptor out rows cols ob orow oc) B p s := rfl

/-- Successful typed-source execution over raw input/output regions and
strides, every native output value and every exterior full memory cell. -/
theorem complete (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) (hB : 0<B) (cover : cols≤B) (hp : p<rows)
    (s : State Word) (values : Nat → Nat → Word)
    (reads : ∀ r c, r<rows → c<cols → (s.mem inp (ib+r*ir+c*ic)).read .real=values r c)
    (injective : ∀ r c r' c', r<rows → c<cols → r'<rows → c'<cols →
      ob+r*orow+c*oc=ob+r'*orow+c'*oc → r=r' ∧ c=c') :
    ∃ t, KernelNative.evaluate inst out inp rows cols ib ob ir ic orow oc B p s=some t ∧
      (∀ c, c<cols → (t.mem out (ob+p*orow+c*oc)).read .real=NativeGraph.row inst 1 cols B (values p) c) ∧
      (∀ r i, (∀ c, c<cols → ¬(out=r ∧ ob+p*orow+c*oc=i)) → t.mem r i=s.mem r i) := by
  let x := KernelNative.descriptor inp rows cols ib ir ic
  let y := KernelNative.descriptor out rows cols ob orow oc
  obtain ⟨t,run⟩ := NativeGraph.source_terminates inst 1 x y B p hB s
  have result := (NativeGraph.source inst 1).row x y B p s t values
    ⟨rfl,rfl,rfl⟩ injective hB cover hp reads run
  refine ⟨t,?_,result.1,?_⟩
  · exact (KernelNative.evaluator_identity inst out inp rows cols ib ob ir ic orow oc B p s).trans run
  · intro r i outside
    apply result.2 r i
    intro q member c hc
    have same : q=p := by simpa using member
    subst q
    exact outside c hc


#axiomsClean complete
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.KernelNative
