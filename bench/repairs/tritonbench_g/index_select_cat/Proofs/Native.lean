import bench.repairs.tritonbench_g.index_select_cat.Contract
import bench.repairs.tritonbench_g.index_select_cat.Proofs.Grid

namespace VeriTile.Bench.Repairs.IndexSelectCat.Native
open Program Grid

theorem checkInputs_iff (r : Request) : checkInputs r=Bool.true ↔ InputContract r := by
  simp only [checkInputs,Bool.and_eq_true,decide_eq_true_eq,InputContract,and_assoc]

theorem index_valid (r : Request) (h : InputContract r) (i : Nat) (hi : i<r.N) (hd : 0<r.D) :
    0 ≤ r.indices i ∧ r.row i<r.M ∧ (r.row i : Int)=r.indices i := by
  have bounds := h.2.2.resolve_left (by omega) ⟨i,hi⟩
  change 0 ≤ r.indices i ∧ r.indices i<(r.M : Int) at bounds
  exact ⟨bounds.1,by simp only [Request.row]; omega,Int.toNat_of_nonneg bounds.1⟩

theorem flat_bound (n d r c : Nat) (hr : r<n) (hc : c<d) : r*d+c<n*d := by nlinarith

/-- All pointer-index expressions fit signed int64 even on inactive column
tails. Only masked-off native pointers may wrap; every active access is
proved inside its actual Tensor window below. -/
theorem all_indices (r : Request) (h : InputContract r) (row ct lane : Nat)
    (hr : row<r.N) (ht : ct<(r.D+511)/512) (hl : lane<512) :
    row<2^31 ∧ ct<65535 ∧ ct*512+lane<2^31 ∧
    r.row row*r.D+(ct*512+lane)<2^63 ∧ row*r.D+(ct*512+lane)<2^63 ∧
    (ct*512+lane<r.D → r.row row*r.D+(ct*512+lane)<r.M*r.D ∧
      row*r.D+(ct*512+lane)<r.N*r.D) := by
  have hd : 0<r.D := by omega
  have valid := index_valid r h row hr hd
  have col : ct*512+lane<r.D+512 := by omega
  rcases h.2.1 with ⟨sourceBytes,outputBytes,indexBytes,rowsLimit,colsLimit⟩
  have sb : r.row row*r.D<r.M*r.D := Nat.mul_lt_mul_of_pos_right valid.2.1 hd
  have ob : row*r.D<r.N*r.D := Nat.mul_lt_mul_of_pos_right hr hd
  refine ⟨by omega,by omega,by omega,by omega,by omega,?_⟩
  intro hc
  exact ⟨flat_bound _ _ _ _ valid.2.1 hc,flat_bound _ _ _ _ hr hc⟩

theorem signed64 (x : Nat) (h : x<2^63) : (BitVec.ofInt 64 (x : Int)).toInt=(x : Int) :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem active_index {α : Type} [Inhabited α] (r : Request) (row ct : Nat)
    (hr : row<r.N) (input output : Nat → α) (lane : Lane) :
    TypedGather.indexOffset (layout (plan r.N r.D r.row row ct))
      (initial (plan r.N r.D r.row row ct) input output) lane<r.N := by
  have zero : lane.1.val=0 := by have bound := lane.1.isLt; omega
  simpa [TypedGather.indexOffset,layout,initial,plan,zero] using hr

theorem source_index {α : Type} [Inhabited α] (r : Request) (h : InputContract r)
    (row ct : Nat) (hr : row<r.N) (hd : 0<r.D) (input output : Nat → α) (lane : Lane) :
    TypedGather.row (layout (plan r.N r.D r.row row ct))
      (initial (plan r.N r.D r.row row ct) input output) lane=r.indices row := by
  have zero : lane.1.val=0 := by have bound := lane.1.isLt; omega
  have same := (index_valid r h row hr hd).2.2
  simpa [TypedGather.row,TypedGather.indexOffset,layout,initial,plan,zero,
    ProgramGrid.sourceRegion,metadataRegion] using same

structure Storage (r : Request) : Prop where
  source : r.source.metadata.dataPtr.toNat+4*(r.M*r.D) ≤ 2^64
  output : r.output.metadata.dataPtr.toNat+4*(r.N*r.D) ≤ 2^64
  index : r.index.metadata.dataPtr.toNat+8*r.N ≤ 2^64

def address (v : View) (bytes i : Nat) : Nat := v.metadata.dataPtr.toNat+bytes*i

theorem pointer (v : View) (bytes count i : Nat)
    (fit : v.metadata.dataPtr.toNat+bytes*count ≤ 2^64) (positive : 0<bytes) (hi : i<count) :
    (BitVec.ofNat 64 (address v bytes i)).toNat=address v bytes i := by
  have bound : address v bytes i<2^64 := by unfold address; nlinarith
  exact Nat.mod_eq_of_lt bound

theorem pointer_aligned (v : View) (bytes i : Nat) (ha : v.metadata.dataPtr.toNat%bytes=0) :
    address v bytes i%bytes=0 := by simp [address,Nat.add_mod,ha]

#axiomsClean checkInputs_iff
#axiomsClean index_valid
#axiomsClean all_indices
#axiomsClean active_index
#axiomsClean source_index
#axiomsClean pointer
end VeriTile.Bench.Repairs.IndexSelectCat.Native
