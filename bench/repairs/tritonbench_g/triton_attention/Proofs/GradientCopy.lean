import bench.repairs.tritonbench_g.triton_attention.Proofs.ByteStores

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native
noncomputable section
open scoped Classical
set_option maxHeartbeats 2500000

/-- Actual incoming gradient coordinates, before the original contiguous copy. -/
def gradientOffset (c : Shape) (incoming : Tensor) (n : Nat) :=
  let group:=n/(c.rows*c.dim)
  (group/c.heads)*incoming.observation.strides[0]?.getD 0+
    (group%c.heads)*incoming.observation.strides[1]?.getD 0+
    ((n/c.dim)%c.rows)*incoming.observation.strides[2]?.getD 0+
    (n%c.dim)*incoming.observation.strides[3]?.getD 0

def gradientWord (c : Shape) (incoming : Tensor) (memory : ByteMemory) (n : Nat) :=
  readWord memory (incoming.dataPointer+4*gradientOffset c incoming n)

def gradientWords (c : Shape) (incoming : Tensor) (memory : ByteMemory) : Memory Word :=
  fun _ n=>gradientWord c incoming memory n

/-- Ordinary bit-preserving Tensor.contiguous semantics, including a possible
identity result. The copy is the one already present in the original source. -/
def copyGradient (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor) (memory : ByteMemory) : ByteMemory :=
  finishBytes c objects [.g] (gradientWords c incoming memory) memory

private theorem copy_separate (objects : Buffer→Tensor) : SeparateWrites objects [.g] := by
  intro r hr s hs hn
  simp only [List.mem_singleton] at hr hs
  exact False.elim (hn (hr.trans hs.symm))

theorem copy_read (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (memory : ByteMemory)
    (hp : pointerChecks g (objects.tensor .g)=Bool.true) (n : Fin (checkedShape g hs).elements) :
    readWord (copyGradient (checkedShape g hs) objects.tensor incoming memory) ((objects.tensor .g).dataPointer+4*n.val)=
      gradientWord (checkedShape g hs) incoming memory n.val := by
  exact finish_word g hs device objects [.g] (by simp)
    (by intro r hr;simp only [List.mem_singleton] at hr;subst r;exact hp)
    (copy_separate objects.tensor) (gradientWords _ incoming memory) memory .g (by simp) n

theorem gradient_offset_contiguous (c : Shape) (incoming : Tensor)
    (h : incoming.observation.strides=Host.strides c) (n : Nat) : gradientOffset c incoming n=n := by
  simp only [gradientOffset,h,Host.strides,List.getElem?_cons_zero,List.getElem?_cons_succ,
    Option.getD_some,Nat.mul_one]
  rw [backward_base_eq_forward]
  have hd : n/(c.rows*c.dim)=(n/c.dim)/c.rows := by rw [Nat.div_div_eq_div_mul,Nat.mul_comm]
  calc
    n/(c.rows*c.dim)*(c.rows*c.dim)+(n/c.dim%c.rows)*c.dim+n%c.dim =
        (n/(c.rows*c.dim)*c.rows+n/c.dim%c.rows)*c.dim+n%c.dim := by ring
    _ = (n/c.dim)*c.dim+n%c.dim := by rw [hd,Nat.div_add_mod']
    _ = n := Nat.div_add_mod' n c.dim

theorem gradient_offset_packed (c : Shape) (incoming : Tensor)
    (h : PackedStrides c incoming) (n : Nat) (hn : n<c.elements) : gradientOffset c incoming n=n := by
  have hg : n/(c.rows*c.dim)<c.batches*c.heads := by
    rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos c.rows_pos c.dim_pos)]
    simpa only [Shape.elements,Shape.groups,Nat.mul_assoc] using hn
  have h0 : n/(c.rows*c.dim)/c.heads*incoming.observation.strides[0]?.getD 0=
      n/(c.rows*c.dim)/c.heads*(c.heads*c.rows*c.dim) := by
    rcases h.1 with he|he
    · have hz : n/(c.rows*c.dim)/c.heads=0 := Nat.div_eq_of_lt (by simpa [he] using hg)
      simp [hz]
    · rw [he]
  have h1 : n/(c.rows*c.dim)%c.heads*incoming.observation.strides[1]?.getD 0=
      n/(c.rows*c.dim)%c.heads*(c.rows*c.dim) := by
    rcases h.2.1 with he|he
    · simp only [he,Nat.mod_one,Nat.zero_mul]
    · rw [he]
  dsimp only [gradientOffset]
  rw [h0,h1,h.2.2.1,h.2.2.2,Nat.mul_one,backward_base_eq_forward]
  have hd : n/(c.rows*c.dim)=(n/c.dim)/c.rows := by rw [Nat.div_div_eq_div_mul,Nat.mul_comm]
  calc
    n/(c.rows*c.dim)*(c.rows*c.dim)+(n/c.dim%c.rows)*c.dim+n%c.dim =
        (n/(c.rows*c.dim)*c.rows+n/c.dim%c.rows)*c.dim+n%c.dim := by ring
    _ = (n/c.dim)*c.dim+n%c.dim := by rw [hd,Nat.div_add_mod']
    _ = n := Nat.div_add_mod' n c.dim

theorem encode_read (memory : ByteMemory) (base : Nat) (j : Fin 4) :
    encode (readWord memory base) j=extendMemory memory (base+j.val) :=
  congrFun (encode_decode (fun i=>extendMemory memory (base+i.val))) j

/-- If contiguous returns the same canonical object, the byte operation is
exactly the identity, including signed zeros and arbitrary NaN payloads. -/
theorem copy_identity (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (memory : ByteMemory)
    (hp : pointerChecks g (objects.tensor .g)=Bool.true) :
    copyGradient (checkedShape g hs) objects.tensor (objects.tensor .g) memory=memory := by
  funext location
  by_cases hex : ∃ i : ByteIndex (checkedShape g hs) [.g],location.val=byteAddress _ objects.tensor [.g] i
  · obtain ⟨i,hi⟩:=hex
    have hr : i.1.val=Buffer.g := by simpa using i.1.property
    have hf:=finish_byte g hs device objects [.g] (by simp)
      (by intro r hr;simp only [List.mem_singleton] at hr;subst r;exact hp)
      (copy_separate objects.tensor) (gradientWords (checkedShape g hs) (objects.tensor .g) memory) memory i
    rw [←hi,extendMemory,dif_pos location.isLt] at hf
    change copyGradient (checkedShape g hs) objects.tensor (objects.tensor .g) memory location=_ at hf
    rw [hf]
    have hn : i.2.1.val<(checkedShape g hs).elements := by simpa only [hr,bufferSize] using i.2.1.isLt
    simp only [gradientWords,gradientWord,gradient_offset_packed _ _ (objects.packed .g (by decide) (by decide)) _ hn,encode_read]
    have he : (objects.tensor .g).dataPointer+4*i.2.1.val+i.2.2.val=location.val := by
      simpa only [byteAddress,hr] using hi.symm
    rw [he]
    exact dif_pos location.isLt
  · apply finish_frame
    intro i h
    exact hex ⟨i,h⟩

theorem copy_preserves_allocation (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming saved : Tensor) (memory : ByteMemory)
    (hp : pointerChecks g (objects.tensor .g)=Bool.true)
    (fresh : Disjoint (objects.tensor .g) saved) (location : Fin (2^64))
    (hlo : saved.storageBase≤location.val) (hhi : location.val < saved.storageBase+saved.storageBytes) :
    copyGradient (checkedShape g hs) objects.tensor incoming memory location=memory location := by
  exact finish_preserves_allocation g hs device objects [.g] (by simp)
    (by intro r hr;simp only [List.mem_singleton] at hr;subst r;exact hp)
    (gradientWords _ incoming memory) memory saved
    (by intro r hr;simp only [List.mem_singleton] at hr;subst r;exact fresh) location hlo hhi

/-- The ordinary contiguous-copy allocation outcome is either the same Tensor
or fresh storage disjoint from the caller's live allocations. -/
def CopyAllocation (objects : Buffer→Tensor) (incoming : Tensor) : Prop :=
  incoming=objects .g ∨ ∀ t∈[incoming,objects .q,objects .k,objects .v,objects .o,objects .l,objects .m],
    Disjoint (objects .g) t

theorem copy_preserves_inputs (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (memory : ByteMemory)
    (hp : pointerChecks g (objects.tensor .g)=Bool.true) (allocation : CopyAllocation objects.tensor incoming)
    (t : Tensor) (ht : t∈[incoming,objects.tensor .q,objects.tensor .k,objects.tensor .v,
      objects.tensor .o,objects.tensor .l,objects.tensor .m]) (location : Fin (2^64))
    (hlo : t.storageBase≤location.val) (hhi : location.val<t.storageBase+t.storageBytes) :
    copyGradient (checkedShape g hs) objects.tensor incoming memory location=memory location := by
  rcases allocation with he|fresh
  · subst incoming
    rw [copy_identity g hs device objects memory hp]
  · exact copy_preserves_allocation g hs device objects incoming t memory hp (fresh t ht) location hlo hhi

#axiomsClean copy_read
#axiomsClean gradient_offset_contiguous
#axiomsClean gradient_offset_packed
#axiomsClean encode_read
#axiomsClean copy_identity
#axiomsClean copy_preserves_allocation
#axiomsClean copy_preserves_inputs
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
