import bench.tritonbench_g.triton_matmul.Proofs.Geometry
import bench.tritonbench_g.triton_matmul.Proofs.Execution
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual quotient/remainder ceiling, with no dimension+block intermediate. -/
def safeCeil (n block : Nat) := n/block + if n%block=0 then 0 else 1
theorem ceiling_normalization (n : Nat) :
    safeCeil n 64=(n+63)/64 ∧ safeCeil n 128=(n+127)/128 ∧
      safeCeil n 256=(n+255)/256 := by
  unfold safeCeil
  split_ifs <;> omega

theorem ceiling_machine (n : Nat) (hn : n<2^63) :
    safeCeil n 64<2^63 ∧ safeCeil n 128<2^63 ∧ safeCeil n 256<2^63 := by
  rw [(ceiling_normalization n).1,(ceiling_normalization n).2.1,(ceiling_normalization n).2.2]
  omega

theorem safeCoordinate_lt (limit x : Nat) (h : 0<limit) : safeCoordinate limit x<limit := by
  unfold safeCoordinate
  split_ifs <;> omega

/-- Clamp both operands before addition, including the padded lane whose
mathematical base+lane may exceed signed64. Neither padded sum is evaluated. -/
theorem clamped_add_machine (limit base lane : Nat) (hp : 0<limit) (hw : limit<2^63) :
    (if lane<limit-base then base else 0)<2^63 ∧
    (if lane<limit-base then lane else 0)<2^63 ∧
    (if lane<limit-base then base else 0)+(if lane<limit-base then lane else 0)<limit := by
  split_ifs <;> omega

theorem nonempty_program (r : Request) (pid : Nat) (hp : pid<programs r) :
    0<r.rows ∧ 0<r.columns := by
  by_cases hr : r.rows=0
  · simp [programs,tileRows,hr] at hp
  by_cases hc : r.columns=0
  · simp [programs,tileColumns,hc] at hp
  omega

theorem control_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) :
    programs r<2^63 ∧ 0<8*tileColumns r ∧ 8*tileColumns r<2^63 ∧
    0<Repair.groupSize (tileRows r) (tileColumns r) 8 pid ∧
    Repair.groupBase (tileRows r) (tileColumns r) 8 pid<tileRows r ∧
    tileRow (tileRows r) (tileColumns r) 8 pid*128<r.rows ∧
    tileCol (tileRows r) (tileColumns r) 8 pid*256<r.columns := by
  obtain ⟨hr,hc⟩ := nonempty_program r pid hp
  obtain ⟨hs,hm,hn⟩ := grouped_bounds (tileRows r) (tileColumns r) 8 pid (by decide) hp
  have htr : tileRows r≤r.rows := by unfold tileRows; omega
  have htc : tileColumns r≤r.columns := by unfold tileColumns; omega
  have hgrid := Nat.mul_le_mul htr htc
  have halloc := t.allocationC
  have hbytes : 1≤elementBytes r.storage := by cases r.storage <;> decide
  have hprod : r.rows*r.columns<2^63 := by nlinarith
  have hbase : Repair.groupBase (tileRows r) (tileColumns r) 8 pid<tileRows r := by
    unfold tileRow at hm
    omega
  have hnwidth := t.dimN
  refine ⟨lt_of_le_of_lt hgrid hprod,?_,?_,hs,hbase,?_,?_⟩
  · unfold tileColumns; omega
  · unfold tileColumns; omega
  · unfold tileRows at hm ⊢; omega
  · unfold tileColumns at hn ⊢; omega

theorem subtraction_machine (r : Request) (pid : Nat) (hp : pid<programs r) :
    (tileRows r : Int)-(Repair.groupBase (tileRows r) (tileColumns r) 8 pid : Nat)=
      (tileRows r-Repair.groupBase (tileRows r) (tileColumns r) 8 pid : Nat) := by
  have h := (grouped_bounds (tileRows r) (tileColumns r) 8 pid (by decide) hp).2.1
  unfold tileRow at h
  omega

theorem inner_base_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (block : Nat) (hb : block<r.parameters.numKBlocks) :
    block*SourceScope.blockK r.storage<r.inner ∧ block*SourceScope.blockK r.storage<2^63 := by
  have hk := t.dimK
  cases hs : r.storage <;>
    simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,hs] at hb ⊢ <;> omega

theorem output_bound (r : Request) (row col : Nat) (hr : row<r.rows) (hc : col<r.columns) :
    outputOffset r row col<r.rows*r.columns := by
  have hn : max r.columns 1=r.columns := by omega
  have hm := Nat.mul_le_mul_left r.columns (Nat.succ_le_of_lt hr)
  unfold outputOffset
  rw [hn]
  nlinarith

theorem safeA_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row k : Nat) (hr : 0<r.rows) (hk : 0<r.inner) : safeAOffset r row k<t.extentA :=
  t.inputA _ (safeCoordinate_lt _ _ hr) _ (safeCoordinate_lt _ _ hk)
theorem safeB_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (k col : Nat) (hk : 0<r.inner) (hc : 0<r.columns) : safeBOffset r k col<t.extentB :=
  t.inputB _ (safeCoordinate_lt _ _ hk) _ (safeCoordinate_lt _ _ hc)
theorem safeC_bound (r : Request) (row col : Nat) (hr : 0<r.rows) (hc : 0<r.columns) :
    safeCOffset r row col<r.rows*r.columns :=
  output_bound r _ _ (safeCoordinate_lt _ _ hr) (safeCoordinate_lt _ _ hc)

theorem safeA_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row k : Nat) (hr : 0<r.rows) (hk : 0<r.inner) :
    safeCoordinate r.rows row*r.strideAM<2^63 ∧ safeCoordinate r.inner k*r.strideAK<2^63 ∧
    safeAOffset r row k<2^63 ∧ elementBytes r.storage*safeAOffset r row k<2^63 ∧
    t.baseA+elementBytes r.storage*safeAOffset r row k<2^64 := by
  have hb := safeA_bound r s t row k hr hk
  have ha := t.allocationA
  have hs := t.spanA
  unfold safeAOffset at *
  cases hh : r.storage <;> simp [hh,elementBytes] at * <;> omega
theorem safeB_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (k col : Nat) (hk : 0<r.inner) (hc : 0<r.columns) :
    safeCoordinate r.inner k*r.strideBK<2^63 ∧ safeCoordinate r.columns col*r.strideBN<2^63 ∧
    safeBOffset r k col<2^63 ∧ elementBytes r.storage*safeBOffset r k col<2^63 ∧
    t.baseB+elementBytes r.storage*safeBOffset r k col<2^64 := by
  have hb := safeB_bound r s t k col hk hc
  have ha := t.allocationB
  have hs := t.spanB
  unfold safeBOffset at *
  cases hh : r.storage <;> simp [hh,elementBytes] at * <;> omega
theorem safeC_machine {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (row col : Nat) (hr : 0<r.rows) (hc : 0<r.columns) :
    max r.columns 1*safeCoordinate r.rows row<2^63 ∧ safeCoordinate r.columns col<2^63 ∧
    safeCOffset r row col<2^63 ∧ elementBytes r.storage*safeCOffset r row col<2^63 ∧
    t.baseC+elementBytes r.storage*safeCOffset r row col<2^64 := by
  have hb := safeC_bound r row col hr hc
  have ha := t.allocationC
  have hs := t.spanC
  unfold safeCOffset at *
  cases hh : r.storage <;> simp [hh,elementBytes] at * <;> omega

theorem candidate_a_pointer {α : Type} (r : Request) (s : State α)
    (block : Nat) (i : AIndex r.parameters) :
    aRow r.parameters s i.1*r.strideAM+
      ((if i.2.1.val<r.inner-block*SourceScope.blockK r.storage then block*SourceScope.blockK r.storage else 0)+
       (if i.2.1.val<r.inner-block*SourceScope.blockK r.storage then i.2.1.val else 0))*r.strideAK=
      safeAOffset r (row r.parameters s i.1) (block*SourceScope.blockK r.storage+i.2.1.val) := by
  rw [Source.clamp_add]
  simp [safeAOffset,safeCoordinate,aRow,Request.parameters,SourceScope.parameters]

theorem candidate_b_pointer {α : Type} (r : Request) (s : State α)
    (block : Nat) (i : BIndex r.parameters) :
    ((if i.1.val<r.inner-block*SourceScope.blockK r.storage then block*SourceScope.blockK r.storage else 0)+
     (if i.1.val<r.inner-block*SourceScope.blockK r.storage then i.1.val else 0))*r.strideBK+
      bCol r.parameters s i.2.1*r.strideBN=
      safeBOffset r (block*SourceScope.blockK r.storage+i.1.val) (col r.parameters s i.2.1) := by
  rw [Source.clamp_add]
  simp [safeBOffset,safeCoordinate,bCol,Request.parameters,SourceScope.parameters]

structure AddressSafety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) : Prop where
  control : ∀ pid, pid<programs r →
    programs r<2^63 ∧ 0<8*tileColumns r ∧ 8*tileColumns r<2^63 ∧
    0<Repair.groupSize (tileRows r) (tileColumns r) 8 pid ∧
    Repair.groupBase (tileRows r) (tileColumns r) 8 pid<tileRows r ∧
    tileRow (tileRows r) (tileColumns r) 8 pid*128<r.rows ∧
    tileCol (tileRows r) (tileColumns r) 8 pid*256<r.columns
  innerBase : ∀ block, block<r.parameters.numKBlocks →
    block*SourceScope.blockK r.storage<r.inner ∧ block*SourceScope.blockK r.storage<2^63
  loadA : ∀ row k, 0<r.rows → 0<r.inner → safeAOffset r row k<t.extentA ∧
    safeAOffset r row k<2^63 ∧ elementBytes r.storage*safeAOffset r row k<2^63 ∧
    t.baseA+elementBytes r.storage*safeAOffset r row k<2^64
  loadB : ∀ k col, 0<r.inner → 0<r.columns → safeBOffset r k col<t.extentB ∧
    safeBOffset r k col<2^63 ∧ elementBytes r.storage*safeBOffset r k col<2^63 ∧
    t.baseB+elementBytes r.storage*safeBOffset r k col<2^64
  store : ∀ row col, 0<r.rows → 0<r.columns → safeCOffset r row col<r.rows*r.columns ∧
    safeCOffset r row col<2^63 ∧ elementBytes r.storage*safeCOffset r row col<2^63 ∧
    t.baseC+elementBytes r.storage*safeCOffset r row col<2^64

theorem address_safety {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) : AddressSafety r s t where
  control := control_machine r s t
  innerBase := inner_base_machine r s t
  loadA := fun row k hr hk => ⟨safeA_bound r s t row k hr hk,(safeA_machine r s t row k hr hk).2.2⟩
  loadB := fun k col hk hc => ⟨safeB_bound r s t k col hk hc,(safeB_machine r s t k col hk hc).2.2⟩
  store := fun row col hr hc => ⟨safeC_bound r row col hr hc,(safeC_machine r s t row col hr hc).2.2⟩

#axiomsClean address_safety
#axiomsClean control_machine
#axiomsClean inner_base_machine
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
