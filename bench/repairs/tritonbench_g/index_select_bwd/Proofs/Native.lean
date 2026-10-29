import bench.repairs.tritonbench_g.index_select_bwd.Contract
import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Program

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Native
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

theorem checkInputs_iff (r : Request) : checkInputs r=Bool.true ↔ InputContract r := by
  simp only [checkInputs,Bool.and_eq_true,decide_eq_true_eq,InputContract,and_assoc]

theorem index_valid (r : Request) (h : InputContract r) (i : Fin r.N) (hc : 0<r.C) :
    0 ≤ r.indices i ∧ (r.indices i).toNat<r.R ∧ ((r.indices i).toNat : Int)=r.indices i := by
  have bounds := h.2.2.resolve_left (by omega) i
  exact ⟨bounds.1,by omega,Int.toNat_of_nonneg bounds.1⟩

/-- No indices are observed in the zero-width branch. This arbitrary valid
row is unobservable because its column type is empty. -/
def selected (r : Request) (h : InputContract r) (i : Fin r.N) : Fin r.R :=
  if hc : r.C=0 then ⟨i.val,lt_of_lt_of_le i.isLt h.1.2.2.2.1⟩
  else ⟨(r.indices i).toNat,(index_valid r h i (by omega)).2.1⟩

theorem selected_value (r : Request) (h : InputContract r) (i : Fin r.N) (c : Fin r.C) :
    (selected r h i).val=(r.indices i).toNat ∧ ((selected r h i).val : Int)=r.indices i := by
  have positive : 0<r.C := by have bound := c.isLt; omega
  simp only [selected,dif_neg (by omega : r.C≠0)]
  exact ⟨True.intro,(index_valid r h i positive).2.2⟩

theorem flat_bound (n d r c : Nat) (hr : r<n) (hc : c<d) : r*d+c<n*d := by nlinarith

/-- Host slices have bounded start/length; masked-off source lanes are
clamped to zero before pointer arithmetic. Each launch has one program. -/
theorem all_indices (r : Request) (h : InputContract r) (i : Fin r.N) (k lane : Nat)
    (hk : k<(r.C+511)/512) (hl : lane<512) :
    0<min 512 (r.C-k*512) ∧ min 512 (r.C-k*512) ≤ 512 ∧
    (let safe := if lane<min 512 (r.C-k*512) then lane else 0
     k*512+safe<r.C ∧
     (selected r h i).val*r.C+(k*512+safe)<r.R*r.C ∧
     i.val*r.C+(k*512+safe)<r.N*r.C ∧
     4*((selected r h i).val*r.C+(k*512+safe))<2^63 ∧
     4*(i.val*r.C+(k*512+safe))<2^63) := by
  have start : k*512<r.C := by omega
  have destinationBytes := h.2.1.1
  have incomingBytes := h.2.1.2.1
  refine ⟨by omega,by omega,?_⟩
  dsimp only
  have hc : k*512+(if lane<min 512 (r.C-k*512) then lane else 0)<r.C := by
    split_ifs <;> omega
  have db := flat_bound r.R r.C _ _ (selected r h i).isLt hc
  have ib := flat_bound r.N r.C _ _ i.isLt hc
  exact ⟨hc,db,ib,by omega,by omega⟩

theorem signed64 (x : Nat) (h : x<2^63) : (BitVec.ofInt 64 (x : Int)).toInt=(x : Int) :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

structure Storage (r : Request) : Prop where
  destination : r.destination.metadata.dataPtr.toNat+4*(r.R*r.C) ≤ 2^64
  incoming : r.incoming.metadata.dataPtr.toNat+4*(r.N*r.C) ≤ 2^64
  index : r.index.metadata.dataPtr.toNat+8*r.N ≤ 2^64

def address (v : View) (bytes i : Nat) : Nat := v.metadata.dataPtr.toNat+bytes*i

theorem pointer (v : View) (bytes count i : Nat)
    (fit : v.metadata.dataPtr.toNat+bytes*count ≤ 2^64) (positive : 0<bytes) (hi : i<count) :
    (BitVec.ofNat 64 (address v bytes i)).toNat=address v bytes i := by
  have bound : address v bytes i<2^64 := by unfold address; nlinarith
  exact Nat.mod_eq_of_lt bound

theorem pointer_aligned (v : View) (bytes i : Nat) (ha : v.metadata.dataPtr.toNat%bytes=0) :
    address v bytes i%bytes=0 := by simp [address,Nat.add_mod,ha]

/-- The exact pointer passed by each direct Python slice plus its source
lane offset equals the caller Tensor's corresponding flat address. -/
theorem slice_address (v : View) (row columns start lane : Nat) :
    address v 4 (row*columns+start)+4*lane=address v 4 (row*columns+(start+lane)) := by
  simp [address,Nat.mul_add,Nat.add_assoc]

theorem source_lane_cast (lane : Nat) (bound : lane<512) :
    (BitVec.ofInt 64 (lane : Int)).toInt=(lane : Int) := signed64 lane (by omega)

#axiomsClean checkInputs_iff
#axiomsClean index_valid
#axiomsClean selected_value
#axiomsClean all_indices
#axiomsClean pointer
#axiomsClean slice_address
#axiomsClean source_lane_cast
end VeriTile.Bench.Repairs.IndexSelectBwd.Native
