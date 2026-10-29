import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Physical
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Bounds

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural FPValue Contract
noncomputable section

/-- The original int32 kernel forms padded offsets before masking. Every
product/sum remains nonnegative and below signed int32, including all 256 VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows. -/
theorem formed (t : Inputs) (s : Scalars) (raw : Nat→Int) (ok : Admitted t s spans raw)
    (b n slot lane k : Nat) (hb : b<t.indices.numel) (hn : n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w)
    (hs : slot<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.w) (hl : lane<256) (hk : k<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).block) :
    xoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b k<2^31 ∧ woff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) slot (n+lane) k<2^31 ∧
      yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n+lane<2^31 := by
  have v:=valid t s spans raw ok
  have bx:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).xm (Nat.le_of_lt hb)
  have bk:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).xk (Nat.le_of_lt hk)
  have ws:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).l0 (Nat.le_of_lt hs)
  have wn:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).wn (Nat.le_of_lt hn)
  have wl:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).wn (Nat.le_of_lt hl)
  have wk:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).wk (Nat.le_of_lt hk)
  have by':=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).ym (Nat.le_of_lt hb)
  have xx:=v.xArithmetic
  have ww:=v.wArithmetic
  have yy:=v.yArithmetic
  simp only [xoff,woff,yoff,Nat.add_mul]
  change _ ∧ _ ∧ b*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y+n+lane<2^31
  change t.indices.numel*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y+VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w+256*1<2^31 at yy
  change b*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y≤t.indices.numel*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y at by'
  dsimp only [VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry] at bx bk ws wn wl wk xx ww ⊢
  omega

theorem signed_formed (t : Inputs) (s : Scalars) (raw : Nat→Int) (ok : Admitted t s spans raw)
    (b n slot lane k : Nat) (hb : b<t.indices.numel) (hn : n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w)
    (hs : slot<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.w) (hl : lane<256) (hk : k<(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).block) :
    (Bounds.xword (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b k).toInt=(xoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b k:Int) ∧
    (Bounds.wword (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) slot (n+lane) k).toInt=(woff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) slot (n+lane) k:Int) ∧
    (Bounds.yword (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n+Bounds.word lane).toInt=(yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n+lane:Int) := by
  have h:=formed t s raw ok b n slot lane k hb hn hs hl hk
  constructor
  · simpa only [Bounds.xword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,xoff] using Bounds.signed_word _ h.1
  constructor
  · simpa only [Bounds.wword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,woff] using Bounds.signed_word _ h.2.1
  · simpa only [Bounds.yword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,yoff,Nat.cast_add] using Bounds.signed_word _ h.2.2

theorem Window.pointer_word {t : TensorInfo} {width : Nat} {backing : StorageRange}
    (w : Window t width backing) (off : Nat) (hi : off<t.numel) (hw : 0<width) :
    (BitVec.ofNat 64 t.dataPtr.toNat+BitVec.ofNat 64 off*BitVec.ofNat 64 width).toNat=
      t.dataPtr.toNat+width*off := by
  have bound:= (w.active_byte off 0 hi hw).2
  simp only [Nat.add_zero] at bound
  simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,Nat.mul_comm,
    Nat.mod_eq_of_lt bound]

theorem addresses (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (b n k : Nat) (hb : b<t.indices.numel) (hn : n<width t.w)
    (hk : k<rank t.w) (active : indices t before b≠-1) :
    (BitVec.ofNat 64 t.x.dataPtr.toNat+BitVec.ofNat 64 (xoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b k)*BitVec.ofNat 64 4).toNat=
      t.x.dataPtr.toNat+4*xoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b k ∧
    (BitVec.ofNat 64 t.w.dataPtr.toNat+BitVec.ofNat 64 (woff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before b).toNat n k)*BitVec.ofNat 64 2).toNat=
      t.w.dataPtr.toNat+2*woff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before b).toNat n k ∧
    (BitVec.ofNat 64 t.y.dataPtr.toNat+BitVec.ofNat 64 (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n := by
  have c:=(python_checks_iff t s spans (indices t before)).mp ok
  have cx : t.x.isContiguous=Bool.true := (c.dense t.x (by
    simp [Contract.tensors,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.tensors])).1
  have cy : t.y.isContiguous=Bool.true := (c.dense t.y (by
    simp [Contract.tensors,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.tensors])).1
  have h:=Bounds.active_bounds (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before) (valid t s spans (indices t before) ok)
    b n k hb hn hk (by simpa [logicalIndex,Contract.geometry,
      VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry] using active)
  simp only [logicalIndex,Contract.geometry,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry,
    Nat.mul_one,Nat.zero_add] at h
  exact ⟨(a.input.window cx).pointer_word _ h.1 (by decide),a.weights.pointer_word _ h.2.1 (by decide),
    (a.output.window cy).pointer_word _ h.2.2 (by decide)⟩

theorem metadata_span (a : Binding t spans) (b : Nat) (hb : b<t.indices.numel)
    (byte : Fin (indexBytes t.indices)) :
    spans.indices.covers (t.indices.dataPtr.toNat+indexBytes t.indices*b+byte.val) :=
  (a.indices.active_byte b byte.val hb byte.isLt).1

#axiomsClean formed
#axiomsClean signed_formed
#axiomsClean Window.pointer_word
#axiomsClean addresses
#axiomsClean metadata_span
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
