import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Physical
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Bounds

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
open VeriTile.Triton FP.Structural Contract
noncomputable section

theorem Window.pointer_word {t : TensorInfo} {width : Nat} {backing : StorageRange}
    (w : Window t width backing) (off : Nat) (hi : off<t.numel) (hw : 0<width) :
    (BitVec.ofNat 64 t.dataPtr.toNat+BitVec.ofNat 64 off*BitVec.ofNat 64 width).toNat=
      t.dataPtr.toNat+width*off := by
  have bound:=(w.active_byte off 0 hi hw).2
  simp only [Nat.add_zero] at bound
  simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,Nat.mul_comm,
    Nat.mod_eq_of_lt bound]

/-- Original clamped coordinates, including masked lanes; the K=0 case uses
output_address and forms no input/weight pointers. -/
theorem addresses (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) (segment : Segment)
    (member : segment∈segments t s (lists t before)) (active : segment.active)
    (hn : 0<(shape t s).columns) (hk : 0<(shape t s).inner)
    (slot : Nat) (selected : segment.adapter=some slot) (r c k : Nat) :
    (BitVec.ofNat 64 t.x.dataPtr.toNat+BitVec.ofNat 64 (Bounds.inputOffset (shape t s) segment r k)*BitVec.ofNat 64 4).toNat=
      t.x.dataPtr.toNat+4*Bounds.inputOffset (shape t s) segment r k ∧
    (BitVec.ofNat 64 t.w.dataPtr.toNat+BitVec.ofNat 64 (Bounds.weightOffset (shape t s) slot c k)*BitVec.ofNat 64 2).toNat=
      t.w.dataPtr.toNat+2*Bounds.weightOffset (shape t s) slot c k ∧
    (BitVec.ofNat 64 t.y.dataPtr.toNat+BitVec.ofNat 64 (Bounds.outputOffset (shape t s) segment r c)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*Bounds.outputOffset (shape t s) segment r c := by
  have contract:=(python_checks_iff t s spans (lists t before)).mp ok
  have cx : t.x.isContiguous=Bool.true := (contract.dense t.x (by simp [tensors])).1
  have cy : t.y.isContiguous=Bool.true := (contract.dense t.y (by simp [tensors])).1
  have h:=Bounds.formed ok segment member active hn hk slot selected r c k
  exact ⟨(a.input.window cx).pointer_word _ h.1 (by decide),
    a.weights.pointer_word _ h.2.1 (by decide),(a.output.window cy).pointer_word _ h.2.2 (by decide)⟩

theorem output_address (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) (segment : Segment)
    (member : segment∈segments t s (lists t before)) (active : segment.active)
    (hn : 0<(shape t s).columns) (r c : Nat) :
    (BitVec.ofNat 64 t.y.dataPtr.toNat+BitVec.ofNat 64 (Bounds.outputOffset (shape t s) segment r c)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*Bounds.outputOffset (shape t s) segment r c := by
  have contract:=(python_checks_iff t s spans (lists t before)).mp ok
  have cy : t.y.isContiguous=Bool.true := (contract.dense t.y (by simp [tensors])).1
  exact (a.output.window cy).pointer_word _ (Bounds.output_formed ok segment member active hn r c) (by decide)

theorem metadata_spans (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) (b : Nat) (hb : b<s.batches.toNat) :
    (∀ byte : Fin (indexBytes t.starts),spans.starts.covers
      (t.starts.dataPtr.toNat+indexBytes t.starts*b+byte.val)) ∧
    (∀ byte : Fin (indexBytes t.lengths),spans.lengths.covers
      (t.lengths.dataPtr.toNat+indexBytes t.lengths*b+byte.val)) ∧
    (∀ byte : Fin (indexBytes t.indices),spans.indices.covers
      (t.indices.dataPtr.toNat+indexBytes t.indices*b+byte.val)) := by
  have c:=(python_checks_iff t s spans (lists t before)).mp ok
  have hs : b<t.starts.numel := by rw [c.shapes.2.2.2.1];exact hb
  have hl : b<t.lengths.numel := hb.trans_le c.shapes.2.2.2.2.2
  have hi : b<t.indices.numel := by rw [c.shapes.2.2.2.2.1];exact hb
  exact ⟨fun byte => (a.starts.active_byte b byte.val hs byte.isLt).1,
    fun byte => (a.lengths.active_byte b byte.val hl byte.isLt).1,
    fun byte => (a.indices.active_byte b byte.val hi byte.isLt).1⟩

#axiomsClean Window.pointer_word
#axiomsClean addresses
#axiomsClean output_address
#axiomsClean metadata_spans
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
