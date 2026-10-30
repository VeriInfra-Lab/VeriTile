import bench.tritonbench_g.isfinite_kernel.Contract
import bench.tritonbench_g.relu_strided_buffer.Proofs.Contract

namespace VeriTile.Bench.Repairs.IsfiniteKernel

theorem contains_iff (start stride : Int) (n : Nat) (address : Int) :
    contains start stride n address = Bool.true ↔ ∃ j,j<n ∧ address=start+(j : Int)*stride := by
  by_cases zero : stride=0
  · simp only [contains,zero,↓reduceIte,decide_eq_true_eq,mul_zero,add_zero]
    constructor
    · rintro ⟨hn,ha⟩; exact ⟨0,hn,ha⟩
    · rintro ⟨j,hj,ha⟩; exact ⟨by omega,ha⟩
  · simpa [contains,zero,ReluStridedBuffer.containsCell] using
      ReluStridedBuffer.containsCell_iff start stride zero n address

theorem point_between (v : View) (n i : Nat) (hi : i<n) :
    min v.pointer (last v.pointer v.byteStride n) ≤ v.signedAddress i ∧
    v.signedAddress i ≤ max v.pointer (last v.pointer v.byteStride n) :=
  ReluStridedBuffer.point_between v.pointer v.byteStride n i hi

theorem span_separated (x y : ViewMetadata) (span : SpanSeparated x y) : PhysicalDisjoint x y := by
  intro i hi k hk j hj equal
  obtain ⟨il,iu⟩ := point_between x.view x.n i hi
  obtain ⟨jl,ju⟩ := point_between y.outputView x.n j hj
  dsimp [SpanSeparated,ViewMetadata.view,ViewMetadata.outputView,View.signedAddress] at span il iu jl ju equal
  rcases span with h | h <;> omega

theorem disjointCheck_iff (x y : ViewMetadata) :
    disjointCheck x y = Bool.true ↔ PhysicalDisjoint x y := by
  have loop : (List.range x.n).all (fun i => (List.range x.width).all
      (fun k => !(contains y.pointer y.stride x.n (x.view.signedAddress i+k)))) = Bool.true ↔
      PhysicalDisjoint x y := by
    simp only [List.all_eq_true,List.mem_range,Bool.not_eq_true']
    constructor
    · intro h i hi k hk j hj equal
      have absent := h i hi k hk
      have present := (contains_iff y.pointer y.stride x.n _).mpr ⟨j,hj,equal⟩
      rw [present] at absent
      contradiction
    · intro separate i hi k hk
      by_contra absent
      have present : contains y.pointer y.stride x.n (x.view.signedAddress i+k) = Bool.true := by
        cases h : contains y.pointer y.stride x.n (x.view.signedAddress i+k) <;> simp_all
      obtain ⟨j,hj,equal⟩ := (contains_iff y.pointer y.stride x.n _).mp present
      exact separate i hi k hk j hj equal
  simp only [disjointCheck,Bool.or_eq_true,decide_eq_true_eq,loop]
  exact ⟨fun h => h.elim (span_separated x y) id,Or.inr⟩

theorem view_contract_iff (out : Bool) (x : ViewMetadata) : ViewContract out x ↔
    x.base.layout = .strided ∧ DTypeValid out x ∧
    (x.device.kind = "cuda" ∧ x.base.device = x.device) ∧
    (x.rawView = Bool.true ∨ x.base.isNeg = Bool.false) ∧
    (x.shape = [x.n] ∧ x.strides = [x.stride]) ∧
    (-(2^63 : Int) ≤ x.stride ∧ x.stride < 2^63) ∧
    (0 < x.n → x.pointer % (x.width : Int) = 0) ∧ (0 < x.n → EndpointBounds x) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.flags,h.rank,h.strideWord,h.alignment,h.bounds⟩
  · rintro ⟨hl,hd,hc,hf,hr,hs,ha,hb⟩; exact ⟨hl,hd,hc,hf,hr,hs,ha,hb⟩

theorem checkView_iff (out : Bool) (x : ViewMetadata) :
    checkView out x = .ok () ↔ ViewContract out x := by
  rw [view_contract_iff]
  cases raw : x.rawView <;> cases neg : x.base.isNeg <;> simp [checkView,raw,neg,ite_eq_iff]

theorem input_contract_iff (x y : ViewMetadata) : InputContract x y ↔
    ViewContract Bool.false x ∧ ViewContract Bool.true y ∧ x.n = y.n ∧ x.device = y.device ∧
    (0 < x.n → y.stride ≠ 0) ∧ PhysicalDisjoint x y := by
  constructor
  · intro h; exact ⟨h.input,h.output,h.shape,h.device,h.outputStride,h.disjoint⟩
  · rintro ⟨hx,hy,hn,hd,hs,ha⟩; exact ⟨hx,hy,hn,hd,hs,ha⟩

theorem checkPair_iff (x y : ViewMetadata) : checkPair x y = .ok () ↔
    x.n = y.n ∧ x.device = y.device ∧ (0 < x.n → y.stride ≠ 0) ∧ PhysicalDisjoint x y := by
  simp [checkPair,disjointCheck_iff,ite_eq_iff]

theorem checkInputs_iff (x y : ArgumentMetadata) : checkInputs x y = .ok () ↔ AdmittedArguments x y := by
  cases x with
  | other => simp [checkInputs,checkArgument,AdmittedArguments,bind,Except.bind]
  | malformed => simp [checkInputs,checkArgument,AdmittedArguments,bind,Except.bind]
  | view x =>
    cases hx : checkView Bool.false x with
    | error e =>
      have bad : ¬ ViewContract Bool.false x := by
        intro h; have := (checkView_iff Bool.false x).mpr h; rw [hx] at this; contradiction
      cases y <;> simp [checkInputs,checkArgument,hx,AdmittedArguments,input_contract_iff,bad,bind,Except.bind]
    | ok u =>
      have good := (checkView_iff Bool.false x).mp hx
      cases y with
      | other => simp [checkInputs,checkArgument,hx,AdmittedArguments,bind,Except.bind]
      | malformed => simp [checkInputs,checkArgument,hx,AdmittedArguments,bind,Except.bind]
      | view y =>
        cases hy : checkView Bool.true y with
        | error e =>
          have bad : ¬ ViewContract Bool.true y := by
            intro h; have := (checkView_iff Bool.true y).mpr h; rw [hy] at this; contradiction
          simp [checkInputs,checkArgument,hx,hy,AdmittedArguments,input_contract_iff,bad,bind,Except.bind]
        | ok u =>
          have goodY := (checkView_iff Bool.true y).mp hy
          simp [checkInputs,checkArgument,hx,hy,AdmittedArguments,input_contract_iff,good,goodY,
            bind,Except.bind,checkPair_iff]

theorem width_pos (v : ViewMetadata) : 0<v.width := by
  unfold ViewMetadata.width; split_ifs <;> norm_num

theorem input_width (v : ViewMetadata) (h : ViewContract Bool.false v) :
    v.precision.width = v.width := by
  have ht := h.dtype
  rcases ht with ht | ht <;> simp [ViewMetadata.precision,ViewMetadata.width,ht,Precision.width]

theorem output_width (v : ViewMetadata) (h : ViewContract Bool.true v) : v.width = 1 := by
  have ht : v.dtype = .other "bool" := h.dtype
  simp [ViewMetadata.width,ht]

theorem accessed_bounds (out : Bool) (v : ViewMetadata) (h : ViewContract out v)
    (i k : Nat) (hi : i<v.n) (hk : k<v.width) :
    (v.storageBase : Int) ≤ v.view.signedAddress i+k ∧
    v.view.signedAddress i+k < v.storageBase+v.capacity ∧
    0 ≤ v.view.signedAddress i ∧ v.view.signedAddress i+k < (2^64 : Nat) := by
  obtain ⟨lo,up⟩ := point_between v.view v.n i hi
  have bounds := h.bounds (by omega)
  dsimp [EndpointBounds,ViewMetadata.view] at bounds lo up ⊢
  norm_cast at *
  omega

theorem output_injective (x y : ViewMetadata) (h : InputContract x y)
    (i j : Nat) (hi : i<x.n) (hj : j<x.n)
    (same : y.outputView.address i = y.outputView.address j) : i=j := by
  have wi := output_width y h.output
  have hn := h.shape
  have iBound := (accessed_bounds Bool.true y h.output i 0 (by omega) (by omega)).2.2.1
  have jBound := (accessed_bounds Bool.true y h.output j 0 (by omega) (by omega)).2.2.1
  have views : y.view = y.outputView := by simp [ViewMetadata.view,ViewMetadata.outputView,wi]
  rw [views] at iBound jBound
  have equality := congrArg Int.ofNat same
  change ((y.outputView.signedAddress i).toNat : Int) = (y.outputView.signedAddress j).toNat at equality
  rw [Int.toNat_of_nonneg iBound,Int.toNat_of_nonneg jBound] at equality
  dsimp [View.signedAddress,ViewMetadata.outputView] at equality
  have product : (i : Int)*y.stride = (j : Int)*y.stride := by omega
  have equal := mul_right_cancel₀ (h.outputStride (by omega)) product
  omega

#axiomsClean checkInputs_iff
#axiomsClean accessed_bounds
#axiomsClean output_injective
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel
