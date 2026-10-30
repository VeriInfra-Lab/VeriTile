import bench.tritonbench_g.kv_cache_filling.Contract

namespace VeriTile.Bench.Repairs.KvCacheFilling

theorem mem_jobs (r : Request) (j : Job) :
    j ∈ jobs r ↔ j.batch < r.batches ∧ j.token < r.count j.batch ∧ j.head < r.heads := by
  cases j
  simp [jobs, Job.mk.injEq, and_assoc, and_left_comm, and_comm]

theorem sourceRow_bound (r : Request) (d : Domain r) (j : Job) (hj : j ∈ jobs r) :
    sourceRow r j < r.tokens * r.heads := by
  obtain ⟨hb, ht, hh⟩ := (mem_jobs r j).mp hj
  have hlen := (d.lengths j.batch hb).2.1
  have htoken : r.start j.batch + j.token < r.tokens := by omega
  unfold sourceRow
  calc
    _ < (r.start j.batch + j.token) * r.heads + r.heads := Nat.add_lt_add_left hh _
    _ = (r.start j.batch + j.token + 1) * r.heads := by ring
    _ ≤ r.tokens * r.heads := Nat.mul_le_mul_right _ (by omega)

theorem destinationToken_bound (r : Request) (d : Domain r) (b t : Nat)
    (hb : b < r.batches) (ht : t < r.count b) : destinationToken r b t < r.pages * r.block := by
  have hp := (d.page_bounds b t hb ht).2
  have hs := Nat.mod_lt (r.history b + t) d.block_pos
  unfold destinationToken
  calc
    _ < r.page b ((r.history b + t) / r.block) * r.block + r.block := Nat.add_lt_add_left hs _
    _ = (r.page b ((r.history b + t) / r.block) + 1) * r.block := by ring
    _ ≤ r.pages * r.block := Nat.mul_le_mul_right _ (by omega)

theorem destinationRow_bound (r : Request) (d : Domain r) (j : Job) (hj : j ∈ jobs r) :
    destinationRow r j < r.pages * r.block * r.heads := by
  obtain ⟨hb, ht, hh⟩ := (mem_jobs r j).mp hj
  have htoken := destinationToken_bound r d j.batch j.token hb ht
  unfold destinationRow
  calc
    _ < destinationToken r j.batch j.token * r.heads + r.heads := Nat.add_lt_add_left hh _
    _ = (destinationToken r j.batch j.token + 1) * r.heads := by ring
    _ ≤ r.pages * r.block * r.heads := Nat.mul_le_mul_right _ (by omega)

theorem destinationRow_injective (r : Request) (d : Domain r) (j j' : Job)
    (hj : j ∈ jobs r) (hj' : j' ∈ jobs r) (he : destinationRow r j = destinationRow r j') : j = j' := by
  obtain ⟨hb, ht, hh⟩ := (mem_jobs r j).mp hj
  obtain ⟨hb', ht', hh'⟩ := (mem_jobs r j').mp hj'
  have hd : destinationToken r j.batch j.token = destinationToken r j'.batch j'.token := by
    have h := congrArg (fun x => x / r.heads) he
    simp only [destinationRow] at h
    rw [Nat.mul_comm (destinationToken r j.batch j.token) r.heads, Nat.mul_add_div d.heads_pos,
      Nat.mul_comm (destinationToken r j'.batch j'.token) r.heads, Nat.mul_add_div d.heads_pos] at h
    simpa [Nat.div_eq_of_lt hh, Nat.div_eq_of_lt hh'] using h
  have ho := d.token_ownership j.batch j.token j'.batch j'.token hb ht hb' ht' hd
  have hhead : j.head = j'.head := by unfold destinationRow at he; rw [hd] at he; exact Nat.add_left_cancel he
  cases j; cases j'; simp_all

def activeRole (r : Request) (role : Role) : Prop :=
  role = .key ∨ role = .value ∨ r.policy ≠ .copy

instance (r : Request) (role : Role) : Decidable (activeRole r role) := inferInstanceAs (Decidable (_ ∨ _ ∨ _))

theorem mem_rowAddresses (r : Request) (j : Job) (a : Address) :
    a ∈ rowAddresses r j ↔ a.row = destinationRow r j ∧ a.column < r.width a.role ∧ activeRole r a.role := by
  rcases a with ⟨role,row,col⟩
  cases role <;> by_cases hp : r.policy = .copy <;>
    simp [rowAddresses, Request.width, activeRole, hp, Address.mk.injEq] <;> omega

theorem row_address_bound (r : Request) (d : Domain r) (j : Job) (hj : j ∈ jobs r)
    (a : Address) (ha : a ∈ rowAddresses r j) :
    a.row < r.pages * r.block * r.heads ∧ a.column < r.width a.role := by
  obtain ⟨he,hc,_⟩ := (mem_rowAddresses r j a).mp ha
  exact ⟨he ▸ destinationRow_bound r d j hj, hc⟩


theorem packed_widths_positive (r : Request) (d : Domain r) : 0 < r.ck ∧ 0 < r.cv := by
  constructor
  · by_contra h
    have hz : r.ck = 0 := by omega
    have hs := d.key_shape
    rw [hz,Nat.mul_zero] at hs
    have hp := d.dk_pos
    omega
  · by_contra h
    have hz : r.cv = 0 := by omega
    have hs := d.value_shape
    rw [hz,Nat.mul_zero] at hs
    have hp := d.dv_pos
    omega

theorem int4_full_head_coordinates (r : Request) (d : Domain r) (hp : r.policy = .int4)
    (i : Nat) (hi : i < r.ck) : i < r.dk ∧ i+r.ck < r.dk := by
  have hs := d.key_shape
  simp [Request.factor,hp] at hs
  omega

theorem int4_value_head_coordinates (r : Request) (d : Domain r) (hp : r.policy = .int4)
    (i : Nat) (hi : i < r.cv) : i < r.dv ∧ i+r.cv < r.dv := by
  have hs := d.value_shape
  simp [Request.factor,hp] at hs
  omega

end VeriTile.Bench.Repairs.KvCacheFilling
