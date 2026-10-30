import bench.repairs.tritonbench_g.relu_strided_buffer.Contract
import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedGrid

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton
open TritonBenchG.ReluStridedBuffer

theorem point_between (start stride : Int) (n i : Nat) (hi : i < n) :
    min start (lastCell start stride n) ≤ start+(i : Int)*stride ∧
    start+(i : Int)*stride ≤ max start (lastCell start stride n) := by
  have bound : (i : Int) ≤ ((n-1 : Nat) : Int) := by omega
  have iz : (0 : Int) ≤ i := by omega
  rcases le_total 0 stride with pos | neg
  · have lo := mul_nonneg iz pos
    have up := mul_le_mul_of_nonneg_right bound pos
    constructor
    · exact le_trans (min_le_left _ _) (by omega)
    · exact le_trans (show start+(i : Int)*stride ≤ lastCell start stride n by
        unfold lastCell; omega) (le_max_right _ _)
  · have lo := mul_le_mul_of_nonpos_right bound neg
    have up := mul_nonpos_of_nonneg_of_nonpos iz neg
    constructor
    · exact le_trans (min_le_right _ _) (show lastCell start stride n ≤ start+(i : Int)*stride by
        unfold lastCell; omega)
    · exact le_trans (by omega) (le_max_left _ _)

theorem endpoint_bounds_iff (x : ViewMetadata) (positive : 0 < x.n) :
    EndpointBounds x ↔ (x.buffer.view x.stride).Valid x.n x.capacity := by
  constructor
  · intro bounds i hi
    obtain ⟨lo,up⟩ := point_between x.buffer.base x.stride x.n i hi
    exact ⟨le_trans bounds.1 lo,lt_of_le_of_lt up bounds.2⟩
  · intro valid
    have first := valid 0 positive
    have last := valid (x.n-1) (by omega)
    change 0 ≤ x.buffer.base+(0 : Int)*x.stride ∧ x.buffer.base+(0 : Int)*x.stride < (x.capacity : Int) at first
    change 0 ≤ lastCell x.buffer.base x.stride x.n ∧ lastCell x.buffer.base x.stride x.n < (x.capacity : Int) at last
    simp only [zero_mul,add_zero] at first
    exact ⟨le_min first.1 last.1,max_lt first.2 last.2⟩

theorem view_contract_iff (x : ViewMetadata) : ViewContract x ↔
    x.base.layout = .strided ∧ (x.dtype = .floating .fp32 ∧ x.base.dtype = .floating .fp32) ∧
    (x.device.kind = "cuda" ∧ x.base.device = x.device) ∧ x.base.isNeg = Bool.false ∧
    (x.buffer.shape = [x.n] ∧ x.buffer.strides = [x.stride]) ∧
    (0 < x.n ∧ x.n < 2^31) ∧ (-(2^63 : Int) ≤ x.stride ∧ x.stride < 2^63) ∧
    x.allocationBase % 4 = 0 ∧ x.pointer = (x.allocationBase : Int)+4*x.buffer.base ∧ EndpointBounds x := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.noNeg,h.rank,h.extent,h.strideWord,h.alignment,h.pointer,h.bounds⟩
  · rintro ⟨hl,hd,hc,hn,hr,he,hs,ha,hp,hb⟩; exact ⟨hl,hd,hc,hn,hr,he,hs,ha,hp,hb⟩

theorem checkView_iff (x : ViewMetadata) : checkView x = .ok () ↔ ViewContract x := by
  rw [view_contract_iff]
  simp [checkView,ite_eq_iff]

theorem containsCell_iff (start stride : Int) (nonzero : stride ≠ 0) (n : Nat) (address : Int) :
    containsCell start stride n address = Bool.true ↔ ∃ j, j < n ∧ address = start+(j : Int)*stride := by
  simp only [containsCell,decide_eq_true_eq]
  constructor
  · rintro ⟨mod,lo,hi⟩
    refine ⟨((address-start)/stride).toNat,by omega,?_⟩
    rw [Int.toNat_of_nonneg lo]
    have product := Int.ediv_mul_cancel_of_emod_eq_zero mod
    omega
  · rintro ⟨j,hj,rfl⟩
    simp only [add_sub_cancel_left,Int.mul_emod_left,Int.mul_ediv_cancel _ nonzero]
    exact ⟨True.intro,by omega,by omega⟩

theorem span_separated (x y : ViewMetadata) (span : SpanSeparated x y) : PhysicalDisjoint x y := by
  intro i hi j hj equal
  obtain ⟨il,iu⟩ := point_between x.start x.stride x.n i hi
  obtain ⟨jl,ju⟩ := point_between y.start y.stride x.n j hj
  rcases span with before | after <;> omega

theorem disjointCheck_iff (x y : ViewMetadata) (nonzero : y.stride ≠ 0) :
    disjointCheck x y = Bool.true ↔ PhysicalDisjoint x y := by
  have loop : (List.range x.n).all
      (fun i => !(containsCell y.start y.stride x.n (x.start+(i : Int)*x.stride))) = Bool.true ↔
      PhysicalDisjoint x y := by
    constructor
    · intro all i hi j hj collision
      have absent := List.all_eq_true.mp all i (List.mem_range.mpr hi)
      have present := (containsCell_iff y.start y.stride nonzero x.n _).mpr ⟨j,hj,collision⟩
      simp [present] at absent
    · intro separate
      apply List.all_eq_true.mpr
      intro i member
      have hi := List.mem_range.mp member
      have absent : containsCell y.start y.stride x.n (x.start+(i : Int)*x.stride) ≠ Bool.true := by
        intro present
        obtain ⟨j,hj,collision⟩ := (containsCell_iff y.start y.stride nonzero x.n _).mp present
        exact separate i hi j hj collision
      cases value : containsCell y.start y.stride x.n (x.start+(i : Int)*x.stride) <;> simp_all

  simp only [disjointCheck,Bool.or_eq_true,decide_eq_true_eq,loop]
  exact ⟨fun h => h.elim (span_separated x y) id,Or.inr⟩

theorem input_contract_iff (x y : ViewMetadata) : InputContract x y ↔
    ViewContract x ∧ ViewContract y ∧ x.n = y.n ∧ x.device = y.device ∧
    y.stride ≠ 0 ∧ PhysicalDisjoint x y := by
  constructor
  · intro h; exact ⟨h.input,h.output,h.shape,h.device,h.outputStride,h.disjoint⟩
  · rintro ⟨hx,hy,hn,hd,hs,hp⟩; exact ⟨hx,hy,hn,hd,hs,hp⟩

theorem checkPair_iff (x y : ViewMetadata) : checkPair x y = .ok () ↔
    x.n = y.n ∧ x.device = y.device ∧ y.stride ≠ 0 ∧ PhysicalDisjoint x y := by
  by_cases hs : y.stride = 0
  · simp [checkPair,hs,ite_eq_iff]
  · simp [checkPair,hs,disjointCheck_iff x y hs,ite_eq_iff]

theorem checkInputs_iff (x y : ArgumentMetadata) : checkInputs x y = .ok () ↔ AdmittedArguments x y := by
  cases x with
  | other => simp [checkInputs,checkArgument,AdmittedArguments,bind,Except.bind]
  | malformed => simp [checkInputs,checkArgument,AdmittedArguments,bind,Except.bind]
  | view x =>
    cases hx : checkView x with
    | error e =>
      have bad : ¬ ViewContract x := by
        intro h; have := (checkView_iff x).mpr h; rw [hx] at this; contradiction
      cases y <;> simp [checkInputs,checkArgument,hx,AdmittedArguments,
        input_contract_iff,bad,bind,Except.bind]
    | ok u =>
      have good := (checkView_iff x).mp hx
      cases y with
      | other => simp [checkInputs,checkArgument,hx,AdmittedArguments,bind,Except.bind]
      | malformed => simp [checkInputs,checkArgument,hx,AdmittedArguments,bind,Except.bind]
      | view y =>
        cases hy : checkView y with
        | error e =>
          have bad : ¬ ViewContract y := by
            intro h; have := (checkView_iff y).mpr h; rw [hy] at this; contradiction
          simp [checkInputs,checkArgument,hx,hy,AdmittedArguments,input_contract_iff,bad,bind,Except.bind]
        | ok u =>
          have goodY := (checkView_iff y).mp hy
          simp [checkInputs,checkArgument,hx,hy,AdmittedArguments,input_contract_iff,good,goodY,
            bind,Except.bind,checkPair_iff]

#axiomsClean endpoint_bounds_iff
#axiomsClean containsCell_iff
#axiomsClean disjointCheck_iff
#axiomsClean checkInputs_iff
end VeriTile.Bench.Repairs.ReluStridedBuffer
