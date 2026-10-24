import bench.repairs.tritonbench_g.rope_transform.Proofs.Host
namespace VeriTile.Bench.Repairs.RopeTransform.ABI
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def last (r : Request) (key : Bool) :=
  (r.rows-1)*r.stride key+(r.paddedHeads key-1)*r.dimension+(r.width-1)+r.half
def coefficientLast (r : Request) := (r.sequence-1)*r.half+(r.width-1)

/-- Input-only checks performed by the wrapper. Masked padded lanes are
included in the formed-address limits. Empty grids never form these pointers. -/
structure Limits (r : Request) : Prop where
  dimensions : r.batches<2^31 ∧ r.sequence<2^31 ∧ r.qHeads<2^31 ∧ r.kHeads<2^31 ∧
    r.dimension<2^31 ∧ r.paddedDimension<2^31 ∧ r.rows<2^31
  heads : ∀ key,r.paddedHeads key<2^31 ∧ r.stride key<2^31
  tile : ∀ key,r.paddedHeads key*r.width≤1048576
  formed : r.rows≠0 → (∀ key,last r key<2^31) ∧ coefficientLast r<2^31

def PointerLimits {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r) : Prop :=
  r.rows≠0 → (∀ key,(call.output key).base+4*last r key+4≤2^64) ∧
    call.cosine.base+4*coefficientLast r+4≤2^64 ∧ call.sine.base+4*coefficientLast r+4≤2^64

theorem formed_output {r : Request} (limits : Limits r) (key second : Bool) (a h j : Nat)
    (ha : a<r.rows) (hh : h<r.paddedHeads key) (hj : j<r.width) :
    r.offset key a h (r.column second j)≤last r key ∧
      r.offset key a h (r.column second j)<2^31 := by
  have h1 := Nat.mul_le_mul_right (r.stride key) (show a≤r.rows-1 by omega)
  have h2 := Nat.mul_le_mul_right r.dimension (show h≤r.paddedHeads key-1 by omega)
  have h3 : j≤r.width-1 := by omega
  have hs : (if second then r.half else 0)≤r.half := by cases second <;> simp
  have ho : r.offset key a h (r.column second j)≤last r key := by
    unfold Request.offset Request.column last; omega
  exact ⟨ho,lt_of_le_of_lt ho ((limits.formed (by omega)).1 key)⟩

theorem formed_coefficient {r : Request} (limits : Limits r) (a j : Nat)
    (ha : a<r.rows) (hj : j<r.width) :
    a%r.sequence*r.half+j≤coefficientLast r ∧ a%r.sequence*r.half+j<2^31 := by
  have hs : 0<r.sequence := by
    by_contra h
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at ha
  have hm := Nat.mod_lt a hs
  have h1 := Nat.mul_le_mul_right r.half (show a%r.sequence≤r.sequence-1 by omega)
  have ho : a%r.sequence*r.half+j≤coefficientLast r := by unfold coefficientLast; omega
  exact ⟨ho,lt_of_le_of_lt ho (limits.formed (by omega)).2⟩

theorem active_output {r : Request} (key second : Bool) (a h j : Nat)
    (ha : a<r.rows) (hh : h<r.heads key) (hj : j<r.half) :
    r.offset key a h (r.column second j)<r.rows*r.stride key := by
  exact lt_of_lt_of_le (offset_window r key a h _ hh (column_lt r second j hj)).2
    (Nat.mul_le_mul_right (r.stride key) (by omega : a+1≤r.rows))

theorem active_coefficient {r : Request} (a j : Nat) (ha : a<r.rows) (hj : j<r.half) :
    a%r.sequence*r.half+j<r.sequence*r.half := by
  have hs : 0<r.sequence := by
    by_contra h
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at ha
  have hm := Nat.mod_lt a hs
  have h1 := Nat.mul_le_mul_right r.half (show a%r.sequence+1≤r.sequence by omega)
  rw [Nat.add_mul,Nat.one_mul] at h1
  omega

theorem active_complete_bytes {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r)
    (key second : Bool) (a h j : Nat) (ha : a<r.rows) (hh : h<r.heads key) (hj : j<r.half) :
    (call.output key).allocationBase≤(call.output key).base+4*r.offset key a h (r.column second j) ∧
    (call.output key).base+4*r.offset key a h (r.column second j)+4≤(call.output key).allocationEnd ∧
    (call.output key).allocationEnd≤2^64 := by
  have ho := active_output key second a h j ha hh hj
  obtain ⟨hb,hend,h64⟩ := call.outputFits key (by omega)
  exact ⟨by omega,by omega,h64⟩

theorem padded_complete_pointer {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r)
    (limits : Limits r) (pointers : PointerLimits r call) (key second : Bool) (a h j : Nat)
    (ha : a<r.rows) (hh : h<r.paddedHeads key) (hj : j<r.width) :
    (call.output key).base+4*r.offset key a h (r.column second j)+4≤2^64 := by
  have ho := (formed_output limits key second a h j ha hh hj).1
  have hp := (pointers (by omega)).1 key
  omega

#axiomsClean formed_output
#axiomsClean formed_coefficient
#axiomsClean active_complete_bytes
#axiomsClean padded_complete_pointer
#specNonCircular Limits avoiding [Host.invoke,Grid.run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RopeTransform.ABI
