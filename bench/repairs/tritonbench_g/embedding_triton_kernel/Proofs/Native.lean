import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Physical

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Native
open VeriTile.Triton Physical Request
noncomputable section

/-- Ordinary valid contiguous Tensor storage: data_ptr is the first accessed
word and its whole window fits the device address space. No kernel result. -/
structure TensorStorage (c : Request) : Prop where
  ids : c.ids.dataPtr.toNat+4*c.ids.numel ≤ 2^64
  weight : c.weight.dataPtr.toNat+4*c.weight.numel ≤ 2^64
  output : c.output.dataPtr.toNat+4*c.output.numel ≤ 2^64

def signed64 (x : Int) : Int := (BitVec.ofInt 64 x).toInt

theorem signed64_exact (x : Int) (lo : -(2^63) ≤ x) (hi : x<2^63) : signed64 x=x :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

def pointer (base element : Nat) : Nat := (BitVec.ofNat 64 (base+4*element)).toNat

theorem pointer_exact (base count i : Nat) (cap : base+4*count ≤ 2^64) (hi : i<count) :
    pointer base i=base+4*i := by
  simp only [pointer,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by omega)

theorem event_bounds (c : Request) (e : Event) (he : e∈events c) :
    e.1<grid c.n ∧ e.2.1<64 ∧ column e<c.hidden.nextPowerOfTwo := by
  simp only [events,List.mem_flatMap,List.mem_map,List.mem_range] at he
  obtain ⟨pid,hp,a,ha,d,hd,eq⟩ := he
  subst e
  exact ⟨hp,ha,hd⟩

theorem rows_exact (c : Request) (h : InputContract c) (e : Event) (he : e∈events c) :
    row e<2^37 ∧ c.n<2^37 ∧
    signed64 ((e.1 : Int)*64)=(e.1 : Int)*64 ∧
    signed64 ((e.1 : Int)*64+e.2.1)=((row e : Nat) : Int) := by
  obtain ⟨hp,ha,_⟩ := event_bounds c e he
  have hg := h.2.2.2.2.1.2.2.2.1
  have hr : row e<2^37 := by unfold row; omega
  have hn : c.n<2^37 := by unfold grid at hg; omega
  refine ⟨hr,hn,signed64_exact _ (by omega) (by omega),?_⟩
  rw [signed64_exact _ (by omega) (by unfold row at hr; omega)]
  simp [row]

theorem normalized_endpoints (c : Request) (h : InputContract c) (hp : 0<c.hidden) :
    -(2^63) ≤ c.firstId ∧ c.firstId<2^63 ∧ -(2^63) ≤ c.lastId ∧ c.lastId<2^63 := by
  have sizes := Bounds.sizes c h
  have hw := h.2.2.2.2.1.2.1
  rw [sizes.2.1] at hw
  have rows : c.rows<2^61 := lt_of_le_of_lt (Nat.le_mul_of_pos_right _ hp) hw
  by_cases inter : c.intersects
  · have vocab := h.2.2.2.2.2.2.2
    have bound : c.upper-c.first ≤ (c.rows : Int) := by
      rcases vocab with hz|hn|hb
      · omega
      · contradiction
      · exact hb
    simp only [firstId,lastId,if_pos inter]
    unfold intersects upper at inter
    unfold upper at bound ⊢
    omega
  · simp only [firstId,lastId,if_neg inter]
    norm_num

/-- ID loads, all actual gather loads and stores lie inside their Tensor
windows. The row and vocabulary masks are proved independently of values. -/
theorem accesses (c : Request) (h : InputContract c) (storage : TensorStorage c)
    (m : Memory) (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    pointer c.ids.dataPtr.toNat r=address c.ids r ∧
    pointer c.output.dataPtr.toNat (r*c.hidden+d)=address c.output (r*c.hidden+d) ∧
    (c.first ≤ idAt c m r ∧ idAt c m r<c.last →
      let i := (idAt c m r-c.first).toNat
      i<c.rows ∧ i*c.hidden+d<c.weight.numel ∧
      pointer c.weight.dataPtr.toNat (i*c.hidden+d)=address c.weight (i*c.hidden+d)) := by
  obtain ⟨hn,hw,ho⟩ := Bounds.sizes c h
  refine ⟨pointer_exact _ _ _ storage.ids (by omega),?_,?_⟩
  · exact pointer_exact _ _ _ storage.output (by rw [ho]; exact Bounds.flat_bound _ _ _ _ hr hd)
  · intro valid
    obtain ⟨lo,hi⟩ := id_bounds c m r
    have ri := Bounds.selected_row c h (by omega) _ lo hi valid
    have bound : (idAt c m r-c.first).toNat*c.hidden+d<c.weight.numel := by
      rw [hw]; exact Bounds.flat_bound _ _ _ _ ri hd
    exact ⟨ri,bound,pointer_exact _ _ _ storage.weight bound⟩

theorem signed_offsets (c : Request) (h : InputContract c) (m : Memory)
    (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    signed64 ((r*c.hidden+d : Nat) : Int)=((r*c.hidden+d : Nat) : Int) ∧
    (c.first ≤ idAt c m r ∧ idAt c m r<c.last →
      signed64 (idAt c m r-c.first)=idAt c m r-c.first ∧
      signed64 ((((idAt c m r-c.first).toNat*c.hidden+d : Nat) : Int))=
        (((idAt c m r-c.first).toNat*c.hidden+d : Nat) : Int)) := by
  obtain ⟨hn,hw,ho⟩ := Bounds.sizes c h
  have ext := h.2.2.2.2.1
  have ob : r*c.hidden+d<c.output.numel := by rw [ho]; exact Bounds.flat_bound _ _ _ _ hr hd
  refine ⟨signed64_exact _ (by omega) (by have := ext.2.2.1; omega),?_⟩
  intro valid
  obtain ⟨lo,hi⟩ := id_bounds c m r
  have ri := Bounds.selected_row c h (by omega) _ lo hi valid
  have wb : (idAt c m r-c.first).toNat*c.hidden+d<c.weight.numel := by
    rw [hw]; exact Bounds.flat_bound _ _ _ _ ri hd
  have rb : (idAt c m r-c.first).toNat ≤ (idAt c m r-c.first).toNat*c.hidden :=
    Nat.le_mul_of_pos_right _ (by omega)
  refine ⟨signed64_exact _ (by omega) (by have := ext.2.1; omega),
    signed64_exact _ (by omega) (by have := ext.2.1; omega)⟩

/-- The finite native event uses actual 64-bit pointers. All signed arithmetic
is exact by rows_exact, normalized_endpoints and signed_offsets. -/
def nativeStep (c : Request) (m : Memory) (e : Event) : Memory :=
  if active c e then
    let id := (m (pointer c.ids.dataPtr.toNat (row e))).toInt
    let value := if c.firstId ≤ id ∧ id<c.lastId then
      m (pointer c.weight.dataPtr.toNat ((id-c.firstId).toNat*c.hidden+column e)) else 0
    Function.update m (pointer c.output.dataPtr.toNat (row e*c.hidden+column e)) value
  else m

theorem native_step_eq (c : Request) (h : InputContract c) (storage : TensorStorage c)
    (m : Memory) (e : Event) : nativeStep c m e=step c m e := by
  by_cases ha : active c e
  · obtain ⟨ids,out,weight⟩ := accesses c h storage m (row e) (column e) ha.1 ha.2
    simp only [nativeStep,step,if_pos ha,ids,out]
    change Function.update m (outputAddress c e)
      (if c.firstId ≤ idAt c m (row e) ∧ idAt c m (row e)<c.lastId then
        m (pointer c.weight.dataPtr.toNat ((idAt c m (row e)-c.firstId).toNat*c.hidden+column e)) else 0)=
      Function.update m (outputAddress c e)
      (if c.firstId ≤ idAt c m (row e) ∧ idAt c m (row e)<c.lastId then
        m (address c.weight ((idAt c m (row e)-c.firstId).toNat*c.hidden+column e)) else 0)
    obtain ⟨lo,hi⟩ := id_bounds c m (row e)
    by_cases valid : c.first ≤ idAt c m (row e) ∧ idAt c m (row e)<c.last
    · have norm := (Bounds.interval c _ lo hi).mpr valid
      rw [if_pos norm,if_pos norm,Bounds.first_eq c _ lo hi valid,(weight valid).2.2]
    · have norm := mt (Bounds.interval c _ lo hi).mp valid
      simp only [if_neg norm]
  · simp [nativeStep,step,ha]

def launch (c : Request) (m : Memory) : Memory := (events c).foldl (nativeStep c) m

theorem launch_eq (c : Request) (h : InputContract c) (storage : TensorStorage c) (m : Memory) :
    launch c m=Physical.launch c m := by
  unfold launch Physical.launch
  congr 1
  funext m e
  exact native_step_eq c h storage m e

def selectedDifference (c : Request) (m : Memory) (e : Event) : Int :=
  (if row e<c.n ∧ c.firstId ≤ idAt c m (row e) ∧ idAt c m (row e)<c.lastId
   then idAt c m (row e) else c.firstId)-c.firstId

/-- Every evaluated widened offset fits, including masked/padded lanes.
Invalid IDs are selected before subtraction; inactive output rows use zero. -/
theorem all_lane_offsets (c : Request) (h : InputContract c) (m : Memory)
    (e : Event) (he : e∈events c) (positive : 0<c.hidden) :
    signed64 (selectedDifference c m e)=selectedDifference c m e ∧
    0 ≤ selectedDifference c m e ∧
    (selectedDifference c m e).toNat*c.hidden+column e<2^63 ∧
    (if row e<c.n then row e else 0)*c.hidden+column e<2^63 := by
  obtain ⟨_,_,hd⟩ := event_bounds c e he
  have extent := h.2.2.2.2.1
  have width := Bounds.nextPower_bound c.hidden extent.2.2.2.2
  have sizes := Bounds.sizes c h
  have weightSize := extent.2.1
  have outputSize := extent.2.2.1
  rw [sizes.2.1] at weightSize
  rw [sizes.2.2] at outputSize
  have diff : 0 ≤ selectedDifference c m e ∧
      (selectedDifference c m e).toNat*c.hidden<2^61 := by
    unfold selectedDifference
    by_cases valid : row e<c.n ∧ c.firstId ≤ idAt c m (row e) ∧ idAt c m (row e)<c.lastId
    · rw [if_pos valid]
      obtain ⟨lo,hi⟩ := id_bounds c m (row e)
      have original := (Bounds.interval c _ lo hi).mp valid.2
      rw [Bounds.first_eq c _ lo hi original]
      have ri := Bounds.selected_row c h positive _ lo hi original
      have prod := Nat.mul_lt_mul_of_pos_right ri positive
      exact ⟨by omega,lt_trans prod weightSize⟩
    · rw [if_neg valid]
      norm_num
  have rb := Nat.le_mul_of_pos_right (selectedDifference c m e).toNat positive
  have out : (if row e<c.n then row e else 0)*c.hidden<2^61 := by
    split_ifs with hr
    · exact lt_trans (Nat.mul_lt_mul_of_pos_right hr positive) outputSize
    · norm_num
  exact ⟨signed64_exact _ (by omega) (by omega),diff.1,by omega,by omega⟩

/-- program_id and arange values are representable before their int64 casts. -/
theorem initial_integer_casts (c : Request) (h : InputContract c) (e : Event) (he : e∈events c) :
    (BitVec.ofInt 32 (e.1 : Int)).toInt=(e.1 : Int) ∧
    (BitVec.ofInt 32 (column e : Int)).toInt=(column e : Int) := by
  obtain ⟨hp,_,hd⟩ := event_bounds c e he
  have extent := h.2.2.2.2.1
  have width := Bounds.nextPower_bound c.hidden extent.2.2.2.2
  have hg := extent.2.2.2.1
  exact ⟨BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega),
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)⟩

theorem address_aligned (t : VeriTile.Torch.TensorMetadata) (h : Request.aligned t)
    (i : Nat) (hi : i<t.numel) : address t i % 4=0 := by
  rcases h with empty|aligned
  · omega
  · simp [address,Nat.add_mod,aligned]

/-- The checked natural alignment applies to every active access. -/
theorem access_alignment (c : Request) (h : InputContract c) (m : Memory)
    (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    address c.ids r % 4=0 ∧ address c.output (r*c.hidden+d) % 4=0 ∧
    (c.first ≤ idAt c m r ∧ idAt c m r<c.last →
      address c.weight ((idAt c m r-c.first).toNat*c.hidden+d) % 4=0) := by
  obtain ⟨sn,sw,so⟩ := Bounds.sizes c h
  have alignment := h.2.2.2.2.2.1
  refine ⟨address_aligned _ alignment.1 _ (by omega),?_,?_⟩
  · apply address_aligned _ alignment.2.2
    rw [so]
    exact Bounds.flat_bound _ _ _ _ hr hd
  · intro valid
    obtain ⟨lo,hi⟩ := id_bounds c m r
    apply address_aligned _ alignment.2.1
    rw [sw]
    exact Bounds.flat_bound _ _ _ _ (Bounds.selected_row c h (by omega) _ lo hi valid) hd

#axiomsClean rows_exact
#axiomsClean normalized_endpoints
#axiomsClean accesses
#axiomsClean signed_offsets
#axiomsClean launch_eq
#axiomsClean all_lane_offsets
#axiomsClean initial_integer_casts
#axiomsClean access_alignment
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Native
