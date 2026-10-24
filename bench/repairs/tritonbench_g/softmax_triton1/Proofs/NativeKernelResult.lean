import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.NativeKernelResult
open VeriTile Triton FP.Structural
noncomputable section

def reference (inst : NativeGraph.Instructions) := NativeGraph.row inst 1

def negateWord (x : NativeGraph.Word) : NativeGraph.Word := x ^^^ (BitVec.ofNat 32 (2^31))

/-- Internal adapter for the existing source evaluator. Only raw region,
shape and stride fields are supplied; Tensor identity/API flags are unused
constants and are not premises of the kernel interface. -/
private def rawBuffer (region : RegionName) (m n offset rs cs : Nat) : Tensor :=
  ⟨0,region,m,n,offset,rs,cs,0,Bool.false⟩

def nativeExecute (inst : NativeGraph.Instructions) (inp out : RegionName)
    (m n ib ob ir ic orow oc B p : Nat) (s : State NativeGraph.Word) : Option (State NativeGraph.Word) :=
  NativeGraph.evaluate inst 1 (rawBuffer inp m n ib ir ic)
    (rawBuffer out m n ob orow oc) B p s

/-- Successful complete native source execution over raw regions, strides,
input words and grid geometry; every active output and all exterior cells.
No Tensor object, API flag or Python admission appears in this interface. -/
theorem softmax_kernel_fp32 (inst : NativeGraph.Instructions) (inp out : RegionName)
    (m n ib ob ir ic orow oc B p : Nat) (hB : 0 < B) (cover : n ≤ B) (hp : p < m)
    (injective : ∀ r c r' c', r<m → c<n → r'<m → c'<n →
      ob+r*orow+c*oc=ob+r'*orow+c'*oc → r=r' ∧ c=c')
    (values : Nat → Nat → NativeGraph.Word) (s : State NativeGraph.Word)
    (input : ∀ r c, r<m → c<n → (s.mem inp (ib+r*ir+c*ic)).read .real=values r c) :
    ∃ t, nativeExecute inst inp out m n ib ob ir ic orow oc B p s=some t ∧
      (∀ c, c<n → (t.mem out (ob+p*orow+c*oc)).read .real=
        reference inst n B (values p) c) ∧
      (∀ r i, (∀ c, c<n → ¬(out=r ∧ ob+p*orow+c*oc=i)) → t.mem r i=s.mem r i) := by
  let x := rawBuffer inp m n ib ir ic
  let y := rawBuffer out m n ob orow oc
  have shape : SameShape x y := ⟨rfl,rfl,rfl⟩
  have inj : y.Injective := injective
  have hi : Inputs Floating.access x values s := input
  obtain ⟨t,he⟩ := NativeGraph.source_terminates inst 1 x y B p hB s
  have result := (NativeGraph.source inst 1).row x y B p s t values shape inj hB cover hp hi he
  refine ⟨t,he,result.1,?_⟩
  intro r i miss
  apply result.2 r i
  intro row hrow c hc
  have same : row=p := by simpa using hrow
  subst row
  exact miss c hc

#axiomsClean softmax_kernel_fp32
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.NativeKernelResult
