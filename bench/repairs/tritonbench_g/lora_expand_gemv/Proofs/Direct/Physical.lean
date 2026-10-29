import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Native
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Bounds

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Direct.NativeCall
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

theorem output_inside (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (indices a before)) (q : Nat) (hq : q<(geometry t s).count) :
    offset (geometry t s) q<t.y.numel := by
  have v:=valid t s (indices a before) ok
  have h:=coordinates (geometry t s) q hq
  exact v.y _ _ h.2 h.1

theorem output_span (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (indices a before)) (q : Nat) (hq : q<(geometry t s).count)
    (owner address : Nat) (covered : (key t s a q).covers owner address) : a.output.backing.covers owner address := by
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,Allocation.key] at oe lo hi
  have inside:=a.output.active_byte (offset (geometry t s) q)
    (address-(a.output.window.baseByte+4*offset (geometry t s) q)) (output_inside t s a before ok q hq) (by omega)
  convert inside.1 using 1 <;> omega

theorem caller_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (indices a before))
    (owner address : Nat) (covered : a.input.backing.covers owner address ∨ a.weights.backing.covers owner address ∨ a.indices.backing.covers owner address) :
    sourceStores rest ops schedule plans t s a before owner address=before owner address := by
  have c:=(python_checks_iff t s (indices a before)).mp ok
  apply source_frame
  intro q hq _ output
  have out:=output_span t s a before ok q hq owner address output
  rcases covered with x|w|i
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.inputSeparate (c.separate t.x (by simp [reads])).symm) owner address x out
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.weightSeparate (c.separate t.w (by simp [reads])).symm) owner address w out
  · exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _
      (a.indexSeparate (c.separate t.indices (by simp [reads])).symm) owner address i out

theorem failure_frame (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (issued : List Nat) (payload : Nat→Raw.Word)
    (actual : ∀ q∈issued,q<(geometry t s).count ∧ indices a before (q/(geometry t s).N)≠-1)
    (owner address : Nat) (outside : ∀ q,q<(geometry t s).count → indices a before (q/(geometry t s).N)≠-1 →
      ¬(key t s a q).covers owner address) :
    stores Raw.ops (key t s a) payload issued before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro q member
  exact outside q (actual q member).1 (actual q member).2

theorem empty (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (zero : t.indices.numel=0 ∨ width t.w=0) : sourceStores rest ops schedule plans t s a before=before := by
  have count : (geometry t s).count=0 := by rcases zero with zero|zero <;> simp [geometry,Geometry.count,zero]
  simp [sourceStores,work,count,stores]

/-- The original int32 kernel forms padded offsets before masking. Every
product/sum remains nonnegative and below signed int32, including all 256 rows. -/
theorem formed (t : Inputs) (s : Scalars) (raw : Nat→Int) (ok : Admitted t s raw)
    (b n slot lane k : Nat) (hb : b<t.indices.numel) (hn : n<width t.w)
    (hs : slot<rows t.w) (hl : lane<256) (hk : k<(geometry t s).block) :
    xoff (geometry t s) b k<2^31 ∧ woff (geometry t s) slot (n+lane) k<2^31 ∧
      yoff (geometry t s) b n+lane<2^31 := by
  have v:=valid t s raw ok
  have bx:=Nat.mul_le_mul_right (geometry t s).xm (Nat.le_of_lt hb)
  have bk:=Nat.mul_le_mul_right (geometry t s).xk (Nat.le_of_lt hk)
  have ws:=Nat.mul_le_mul_right (geometry t s).l0 (Nat.le_of_lt hs)
  have wn:=Nat.mul_le_mul_right (geometry t s).wn (Nat.le_of_lt hn)
  have wl:=Nat.mul_le_mul_right (geometry t s).wn (Nat.le_of_lt hl)
  have wk:=Nat.mul_le_mul_right (geometry t s).wk (Nat.le_of_lt hk)
  have by':=Nat.mul_le_mul_right (geometry t s).ym (Nat.le_of_lt hb)
  have xx:=v.xArithmetic
  have ww:=v.wArithmetic
  have yy:=v.yArithmetic
  simp only [xoff,woff,yoff,Nat.add_mul]
  change _ ∧ _ ∧ b*columns t.y+n+lane<2^31
  change t.indices.numel*columns t.y+width t.w+256*1<2^31 at yy
  change b*columns t.y≤t.indices.numel*columns t.y at by'
  dsimp only [geometry] at bx bk ws wn wl wk xx ww ⊢
  omega

theorem signed_formed (t : Inputs) (s : Scalars) (raw : Nat→Int) (ok : Admitted t s raw)
    (b n slot lane k : Nat) (hb : b<t.indices.numel) (hn : n<width t.w)
    (hs : slot<rows t.w) (hl : lane<256) (hk : k<(geometry t s).block) :
    (Bounds.xword (geometry t s) b k).toInt=(xoff (geometry t s) b k:Int) ∧
    (Bounds.wword (geometry t s) slot (n+lane) k).toInt=(woff (geometry t s) slot (n+lane) k:Int) ∧
    (Bounds.yword (geometry t s) b n+Bounds.word lane).toInt=(yoff (geometry t s) b n+lane:Int) := by
  have h:=formed t s raw ok b n slot lane k hb hn hs hl hk
  constructor
  · simpa only [Bounds.xword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,xoff] using Bounds.signed_word _ h.1
  constructor
  · simpa only [Bounds.wword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,woff] using Bounds.signed_word _ h.2.1
  · simpa only [Bounds.yword,Bounds.word,←BitVec.ofNat_mul,←BitVec.ofNat_add,yoff,Nat.cast_add] using Bounds.signed_word _ h.2.2

theorem addresses (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (indices a before)) (b n k : Nat) (hb : b<t.indices.numel) (hn : n<width t.w)
    (hk : k<rank t.w) (active : indices a before b≠-1) :
    (BitVec.ofNat 64 a.input.window.baseByte+BitVec.ofNat 64 (xoff (geometry t s) b k)*BitVec.ofNat 64 4).toNat=
      t.x.dataPtr.toNat+4*xoff (geometry t s) b k ∧
    (BitVec.ofNat 64 a.weights.window.baseByte+BitVec.ofNat 64 (woff (geometry t s) (indices a before b).toNat n k)*BitVec.ofNat 64 2).toNat=
      t.w.dataPtr.toNat+2*woff (geometry t s) (indices a before b).toNat n k ∧
    (BitVec.ofNat 64 a.output.window.baseByte+BitVec.ofNat 64 (yoff (geometry t s) b n)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*yoff (geometry t s) b n := by
  have h:=Bounds.active_bounds (geometry t s) (indices a before) (valid t s (indices a before) ok) b n k hb hn hk
    (by simpa [logicalIndex,geometry] using active)
  simp only [logicalIndex,geometry,Nat.mul_one,Nat.zero_add] at h
  exact ⟨a.input.pointer_word _ h.1 (by decide),a.weights.pointer_word _ h.2.1 (by decide),a.output.pointer_word _ h.2.2 (by decide)⟩

theorem metadata_span (a : Binding t) (b : Nat) (hb : b<t.indices.numel)
    (byte : Fin (VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.indexBytes t.indices)) :
    a.indices.backing.covers a.indices.window.owner
      (a.indices.window.baseByte+VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.indexBytes t.indices*b+byte.val) :=
  (a.indices.active_byte b byte.val hb byte.isLt).1

structure Result (y : TensorInfo) where
  tensor : TensorInfo
  same : tensor=y
  returned : Unit
  bytes : Raw.State

def invoke (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (_ok : Admitted t s (indices a before)) : Result t.y :=
  ⟨t.y,rfl,(),sourceStores rest ops schedule plans t s a before⟩

#axiomsClean output_inside
#axiomsClean caller_frame
#axiomsClean failure_frame
#axiomsClean empty
#axiomsClean formed
#axiomsClean signed_formed
#axiomsClean addresses
#axiomsClean metadata_span
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Direct.NativeCall
