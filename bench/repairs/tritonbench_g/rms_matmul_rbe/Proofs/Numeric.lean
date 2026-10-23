import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Events

namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 5000000

/-- Precisely local numeric leaves. Their reference meaning and actual SM90
instruction/library correspondence are jointly pending. This record has no
source K, tensor, reduction, output, memory, execution or frame callback. -/
structure NumericLeaves (actual reference : StorageOps (BitVec 32) (BitVec 16)) : Prop where
  scalar : actual.toOps.Hom reference.toOps id
  hzero : actual.hzero=reference.hzero
  widen : ∀ v,actual.widen v=reference.widen v
  narrow : ∀ v,actual.narrow v=reference.narrow v
  e5Half : ∀ v,actual.e5Half v=reference.e5Half v
  e5Single : ∀ v,actual.e5Single v=reference.e5Single v
  e5Encode : ∀ v,actual.e5Encode v=reference.e5Encode v

theorem Ops.Hom.identity (a b : Ops F) (h : a.Hom b id) : a=b := by
  have h_zero : a.zero=b.zero := h.zero
  have h_one : a.one=b.one := h.one
  have h_add : a.add=b.add := funext (fun x => funext (fun y => h.add x y))
  have h_mul : a.mul=b.mul := funext (fun x => funext (fun y => h.mul x y))
  have h_mulFTZ : a.mulFTZ=b.mulFTZ := funext (fun x => funext (fun y => h.mulFTZ x y))
  have h_divide : a.divide=b.divide := funext (fun x => funext (fun y => h.divide x y))
  have h_rsqrt : a.rsqrt=b.rsqrt := funext (fun x => h.rsqrt x)
  have h_neg : a.neg=b.neg := funext (fun x => h.neg x)
  have h_fma : a.fma=b.fma := funext (fun x => funext (fun y => funext (fun z => h.fma x y z)))
  have h_ofInt : a.ofInt=b.ofInt := funext (fun x => h.ofInt x)
  have h_pow : a.pow=b.pow := funext (fun x => funext (fun y => h.pow x y))
  have h_cos : a.cos=b.cos := funext (fun x => h.cos x)
  have h_sin : a.sin=b.sin := funext (fun x => h.sin x)
  have h_mma8 : a.mma8=b.mma8 := funext (fun x => funext (fun y => funext (fun z => h.mma8 x y z)))
  cases a
  cases b
  simp_all

theorem NumericLeaves.identity (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (h : NumericLeaves actual reference) : actual=reference := by
  have hs := Ops.Hom.identity actual.toOps reference.toOps h.scalar
  have h0 := h.hzero
  have h1 := funext h.widen
  have h2 := funext h.narrow
  have h3 := funext h.e5Half
  have h4 := funext h.e5Single
  have h5 := funext h.e5Encode
  cases actual
  cases reference
  simp_all

theorem native_word_and_frame (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference) (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (sep : b.Separate)
    (injective : Function.Injective (outputAddress b : Cell s → Address))
    (plans : Fin 3 → Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Fin 3 → Nat → Nat → Reduction) (schedule : Schedule s) (state : σ) :
    (∀ x : Cell s,abi.observe ((gridCode p b plans reductions schedule.events).native abi state).2
        (outputAddress b x)=wordTarget reference p b plans reductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((gridCode p b plans reductions schedule.events).native abi state).2 q=abi.observe state q) := by
  have eq := leaves.identity
  subst reference
  rw [native_grid]
  exact whole_word_and_frame actual p b sep injective plans reductions schedule (abi.observe state)

#axiomsClean Ops.Hom.identity
#axiomsClean NumericLeaves.identity
#axiomsClean native_word_and_frame
end
end VeriTile.Bench.Repairs.RmsMatmulRbe
