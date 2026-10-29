import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Store

namespace VeriTile.Bench.Repairs.KvCacheFilling

def elementBytes (r : Request) : Role → Nat
  | .key | .value => if r.policy = .copy then 4 else 1
  | .keyMeta | .valueMeta => 4

/-- Ordinary actual contiguous-tensor/allocator descriptors: base pointer,
backing bytes, byte storage offset, capacity and separate allocations.
There is no final-state, successful execution or numerical assumption. -/
structure Layout (r : Request) where
  base : Role → Nat
  size : Role → Nat
  offset : Role → Nat
  capacity : ∀ role, activeRole r role →
    offset role + r.pages * r.block * r.heads * r.width role * elementBytes r role ≤ size role
  separated : ∀ a b, activeRole r a → activeRole r b → a ≠ b →
    base a + size a ≤ base b ∨ base b + size b ≤ base a

def bytePtr (r : Request) (l : Layout r) (a : Address) (b : Nat) : Nat :=
  l.base a.role + l.offset a.role + (a.row * r.width a.role + a.column) * elementBytes r a.role + b

def ValidByte (r : Request) (a : Address) (b : Nat) : Prop :=
  a.row < r.pages * r.block * r.heads ∧ a.column < r.width a.role ∧
    activeRole r a.role ∧ b < elementBytes r a.role

theorem flat_bound (row col rows width : Nat) (hr : row < rows) (hc : col < width) :
    row * width + col < rows * width := by
  calc
    _ < row * width + width := Nat.add_lt_add_left hc _
    _ = (row+1)*width := by ring
    _ ≤ rows*width := Nat.mul_le_mul_right _ (by omega)

theorem flat_injective (row row' col col' width : Nat) (hc : col < width) (hc' : col' < width)
    (he : row*width+col = row'*width+col') : row=row' ∧ col=col' := by
  have hp : 0 < width := by omega
  have hd := congrArg (fun i => i / width) he
  dsimp only at hd
  rw [Nat.mul_comm row width,Nat.mul_add_div hp,Nat.mul_comm row' width,Nat.mul_add_div hp] at hd
  have hr : row = row' := by simpa [Nat.div_eq_of_lt hc,Nat.div_eq_of_lt hc'] using hd
  subst row'
  exact ⟨rfl,Nat.add_left_cancel he⟩

theorem bytePtr_bound (r : Request) (l : Layout r) (a : Address) (b : Nat) (h : ValidByte r a b) :
    l.base a.role ≤ bytePtr r l a b ∧ bytePtr r l a b < l.base a.role + l.size a.role := by
  obtain ⟨hr,hc,ha,hb⟩ := h
  have hflat := flat_bound a.row a.column _ _ hr hc
  have hbyte := flat_bound (a.row*r.width a.role+a.column) b _ _ hflat hb
  have hcap := l.capacity a.role ha
  unfold bytePtr
  omega

theorem bytePtr_injective (r : Request) (l : Layout r) (a a' : Address) (b b' : Nat)
    (h : ValidByte r a b) (h' : ValidByte r a' b')
    (he : bytePtr r l a b = bytePtr r l a' b') : a=a' ∧ b=b' := by
  have ha := bytePtr_bound r l a b h
  have ha' := bytePtr_bound r l a' b' h'
  have hr : a.role = a'.role := by
    by_contra hn
    rcases l.separated a.role a'.role h.2.2.1 h'.2.2.1 hn with hs | hs <;> omega
  have hf : (a.row*r.width a.role+a.column)*elementBytes r a.role+b =
      (a'.row*r.width a.role+a'.column)*elementBytes r a.role+b' := by
    unfold bytePtr at he
    rw [← hr] at he
    omega
  obtain ⟨hcell,hb⟩ := flat_injective _ _ _ _ _ h.2.2.2 (by simpa [hr] using h'.2.2.2) hf
  obtain ⟨hrow,hcol⟩ := flat_injective _ _ _ _ _ h.2.1 (by simpa [hr] using h'.2.1) hcell
  exact ⟨by cases a; cases a'; simp_all,hb⟩

abbrev Byte := BitVec 8

def physical (p : Nat) : Address := ⟨.key,p,0⟩

def cellByte {α : Type} (encode : α → Nat → Byte) : Cell α Byte → Nat → Byte
  | .fp32 x,b => encode x b
  | .uint8 x,_ => x

def expandStore {α : Type} (r : Request) (l : Layout r) (encode : α → Nat → Byte)
    (s : Store (Cell α Byte)) : List (Store Byte) :=
  (List.range (elementBytes r s.address.role)).map fun b =>
    ⟨physical (bytePtr r l s.address b), cellByte encode s.value b⟩

def byteStores {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) :=
  (stores r ops kp vp k v).flatMap (expandStore r l encode)

def executeBytes {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) (m : Nat → Byte) : Nat → Byte :=
  fun p => runStores (byteStores r l ops kp vp k v encode) (fun a => m a.row) (physical p)

theorem mem_expandStore {α : Type} (r : Request) (l : Layout r) (encode : α → Nat → Byte)
    (s : Store (Cell α Byte)) (t : Store Byte) :
    t ∈ expandStore r l encode s ↔ ∃ b, b < elementBytes r s.address.role ∧
      t.address = physical (bytePtr r l s.address b) ∧ t.value = cellByte encode s.value b := by
  simp only [expandStore,List.mem_map,List.mem_range]
  constructor
  · rintro ⟨b,hb,ht⟩; subst t; exact ⟨b,hb,rfl,rfl⟩
  · rintro ⟨b,hb,ha,hv⟩; exact ⟨b,hb,by cases t; simp_all⟩

theorem whole_bytes_readback {α : Type} (r : Request) (d : Domain r) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) (m : Nat → Byte)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) :
    executeBytes r l ops kp vp k v encode m (bytePtr r l a b) = cellByte encode (taskValue r ops kp vp k v j a) b := by
  apply runStores_readback
  · let s : Store (Cell α Byte) := ⟨a,taskValue r ops kp vp k v j a⟩
    refine ⟨⟨physical (bytePtr r l a b),cellByte encode s.value b⟩,?_,rfl⟩
    apply List.mem_flatMap.mpr
    refine ⟨s,?_,(mem_expandStore ..).mpr ⟨b,hb,rfl,rfl⟩⟩
    exact List.mem_flatMap.mpr ⟨j,hj,(mem_taskStores ..).mpr ⟨ha,rfl⟩⟩
  · intro t ht he
    obtain ⟨s,hs,ht'⟩ := List.mem_flatMap.mp ht
    obtain ⟨b',hb',hta,htv⟩ := (mem_expandStore ..).mp ht'
    obtain ⟨j',hj',hs'⟩ := List.mem_flatMap.mp hs
    obtain ⟨ha',hsv⟩ := (mem_taskStores ..).mp hs'
    have va : ValidByte r a b := ⟨(row_address_bound r d j hj a ha).1,
      (mem_rowAddresses r j a).mp ha |>.2.1,(mem_rowAddresses r j a).mp ha |>.2.2,hb⟩
    have va' : ValidByte r s.address b' := ⟨(row_address_bound r d j' hj' s.address ha').1,
      (mem_rowAddresses r j' s.address).mp ha' |>.2.1,(mem_rowAddresses r j' s.address).mp ha' |>.2.2,hb'⟩
    have hp : bytePtr r l s.address b' = bytePtr r l a b := by
      have h := congrArg Address.row (hta.symm.trans he)
      exact h
    obtain ⟨hsa,hbb⟩ := bytePtr_injective r l s.address a b' b va' va hp
    have hsame : j'=j := destinationRow_injective r d j' j hj' hj (by
      rw [← ((mem_rowAddresses r j' s.address).mp ha').1,
        ← ((mem_rowAddresses r j a).mp ha).1,hsa])
    simpa [hsa,hbb,hsame,hsv] using htv

theorem whole_bytes_frame {α : Type} (r : Request) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) (m : Nat → Byte)
    (p : Nat) (h : ∀ j ∈ jobs r, ∀ a ∈ rowAddresses r j, ∀ b, b < elementBytes r a.role → bytePtr r l a b ≠ p) :
    executeBytes r l ops kp vp k v encode m p = m p := by
  apply runStores_frame
  intro t ht he
  obtain ⟨s,hs,ht'⟩ := List.mem_flatMap.mp ht
  obtain ⟨b,hb,ha,_⟩ := (mem_expandStore ..).mp ht'
  obtain ⟨j,hj,hs'⟩ := List.mem_flatMap.mp hs
  exact h j hj s.address ((mem_taskStores ..).mp hs').1 b hb (congrArg Address.row (ha.symm.trans he))

/-- Prefix/suffix backing bytes, unrelated allocations and input allocations
are preserved whenever their addresses lie outside active output allocation spans. -/
theorem outside_allocations_frame {α : Type} (r : Request) (d : Domain r) (l : Layout r) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (encode : α → Nat → Byte) (m : Nat → Byte)
    (p : Nat) (h : ∀ role, activeRole r role → p < l.base role ∨ l.base role + l.size role ≤ p) :
    executeBytes r l ops kp vp k v encode m p = m p := by
  apply whole_bytes_frame
  intro j hj a ha b hb he
  have ha' := (mem_rowAddresses r j a).mp ha
  have hbound := bytePtr_bound r l a b ⟨(row_address_bound r d j hj a ha).1,ha'.2.1,ha'.2.2,hb⟩
  rcases h a.role ha'.2.2 with hp | hp <;> omega


theorem distinct_jobs_disjoint (r : Request) (d : Domain r) (l : Layout r)
    (j j' : Job) (hj : j ∈ jobs r) (hj' : j' ∈ jobs r) (hne : j ≠ j')
    (a a' : Address) (ha : a ∈ rowAddresses r j) (ha' : a' ∈ rowAddresses r j')
    (b b' : Nat) (hb : b < elementBytes r a.role) (hb' : b' < elementBytes r a'.role) :
    bytePtr r l a b ≠ bytePtr r l a' b' := by
  intro he
  have hi := (mem_rowAddresses r j a).mp ha
  have hi' := (mem_rowAddresses r j' a').mp ha'
  have hab := (bytePtr_injective r l a a' b b'
    ⟨(row_address_bound r d j hj a ha).1,hi.2.1,hi.2.2,hb⟩
    ⟨(row_address_bound r d j' hj' a' ha').1,hi'.2.1,hi'.2.2,hb'⟩ he).1
  exact hne (destinationRow_injective r d j j' hj hj' (by rw [← hi.1,← hi'.1,hab]))

end VeriTile.Bench.Repairs.KvCacheFilling
