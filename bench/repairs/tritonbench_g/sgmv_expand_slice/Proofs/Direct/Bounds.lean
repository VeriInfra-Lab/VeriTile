import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Common
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Integer

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.Bounds
open Contract
noncomputable section
set_option maxHeartbeats 4000000

def inputOffset (p : Shape) (segment : Segment) (r k : Nat) :=
  (segment.start+r%segment.length)*p.inner+safeInner p k
def weightOffset (p : Shape) (slot c k : Nat) :=
  slot*(p.columns*p.inner)+(c%p.columns)*p.inner+safeInner p k
def outputOffset (p : Shape) (segment : Segment) (r c : Nat) :=
  safeRow segment r*p.outputColumns+safeColumn p c

/-- Clamped loads and stores stay in their original views, including masked
lanes. K=0 forms no input/weight pointers and still stores valid output. -/
theorem formed (ok : Admitted t s lists) (segment : Segment)
    (member : segment∈segments t s lists) (active : segment.active)
    (hn : 0<(shape t s).columns) (hk : 0<(shape t s).inner)
    (slot : Nat) (selected : segment.adapter=some slot) (r c k : Nat) :
    inputOffset (shape t s) segment r k<t.x.numel ∧
    weightOffset (shape t s) slot c k<t.w.numel ∧
    outputOffset (shape t s) segment r c<t.y.numel := by
  let p:=shape t s
  have fit:=fitted ok segment member
  have xr:=gathered_row_in_bounds p segment fit active r
  have wc:=gathered_column_in_bounds p hn c
  have ik:=safeInner_lt p hk k
  have wa:=fit.adapter slot selected active.1
  rw [input_numel ok,weight_numel ok,output_numel ok]
  refine ⟨linear_lt xr ik,?_,clamped_output_offset p segment fit active (capacity ok) hn r c⟩
  have inner:=linear_lt wc ik
  have all:=linear_lt wa inner
  simpa only [weightOffset,Nat.add_assoc] using all

theorem output_formed (ok : Admitted t s lists) (segment : Segment)
    (member : segment∈segments t s lists) (active : segment.active)
    (hn : 0<(shape t s).columns) (r c : Nat) : outputOffset (shape t s) segment r c<t.y.numel := by
  rw [output_numel ok]
  exact clamped_output_offset _ _ (fitted ok segment member) active (capacity ok) hn r c

theorem signed_bytes (ok : Admitted t s lists) (segment : Segment)
    (member : segment∈segments t s lists) (active : segment.active)
    (hn : 0<(shape t s).columns) (hk : 0<(shape t s).inner)
    (slot : Nat) (selected : segment.adapter=some slot) (r c k : Nat) :
    4*inputOffset (shape t s) segment r k<2^63 ∧
    2*weightOffset (shape t s) slot c k<2^63 ∧
    4*outputOffset (shape t s) segment r c<2^63 := by
  have h:=formed ok segment member active hn hk slot selected r c k
  have e:=((python_checks_iff t s lists).mp ok).extents
  dsimp only [Extents] at e
  omega

theorem signed_offsets (ok : Admitted t s lists) (segment : Segment)
    (member : segment∈segments t s lists) (active : segment.active)
    (hn : 0<(shape t s).columns) (hk : 0<(shape t s).inner)
    (slot : Nat) (selected : segment.adapter=some slot) (r c k : Nat) :
    (Integer.i64 ((segment.start+r%segment.length)*(shape t s).inner)+Integer.i64 (safeInner (shape t s) k)).toInt=
      (inputOffset (shape t s) segment r k:Int) ∧
    (Integer.i64 slot*Integer.i64 ((shape t s).columns*(shape t s).inner)+
      Integer.i64 (c%(shape t s).columns)*Integer.i64 (shape t s).inner+Integer.i64 (safeInner (shape t s) k)).toInt=
      (weightOffset (shape t s) slot c k:Int) ∧
    (Integer.i64 (safeRow segment r)*Integer.i64 (shape t s).outputColumns+Integer.i64 (safeColumn (shape t s) c)).toInt=
      (outputOffset (shape t s) segment r c:Int) := by
  have h:=signed_bytes ok segment member active hn hk slot selected r c k
  dsimp only [inputOffset,weightOffset,outputOffset] at h
  simp only [Integer.i64,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact ⟨Integer.i64_value (by omega),Integer.i64_value (by omega),Integer.i64_value (by omega)⟩

#axiomsClean formed
#axiomsClean output_formed
#axiomsClean signed_bytes
#axiomsClean signed_offsets
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.Bounds
