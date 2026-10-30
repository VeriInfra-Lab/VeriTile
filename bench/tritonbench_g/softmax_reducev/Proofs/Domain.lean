import bench.tritonbench_g.softmax_reducev.Proofs.Host
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked
open VeriTile.Triton FP.Structural
open Host
noncomputable section
set_option maxHeartbeats 1000000

def pageOffset (c : Call) (x : Input α) (j : Job) (k : Nat) :=
  j.batch*c.maximum+(c.maximum-x.counts j.batch)+k

def logicAddress (c : Call) (x : Input α) (j : Job) (k : Nat) :=
  j.head*c.tokens+x.starts j.batch+k

def valueAddress (c : Call) (x : Input α) (j : Job) (k d : Nat) :=
  (x.pages (pageOffset c x j k)*c.heads+j.head)*c.dimensions+d

/-- The admitted native decoding contract. Tensor shape/dtype/contiguous and
signed-int32 extent checks are in the Python wrapper. Nonempty metadata,
valid active page indices, and nonaliasing of allocations are caller facts. -/
structure Admissible (c : Call) (x : Input α) : Prop where
  dimension : ∃ k, c.dimensions=2^k
  maximum : 0 < c.maximum
  maximum_width : c.maximum ≤ 2^31-64
  logits_width : c.heads*c.tokens ≤ 2^31-64
  values_width : c.pages*c.heads*c.dimensions ≤ 2^31-64
  output_width : c.batch*c.heads*c.dimensions ≤ 2^31-64
  locations_width : c.batch*c.maximum ≤ 2^31-64
  lengths : ∀ b, b < c.batch → 0 < x.counts b ∧ x.counts b ≤ c.maximum ∧ x.starts b+x.counts b ≤ c.tokens
  pages : ∀ j, Legal c j → ∀ k, k < x.counts j.batch → x.pages (pageOffset c x j k) < c.pages

theorem flat_lt (i j n m : Nat) (hi : i < n) (hj : j < m) : i*m+j < n*m := by
  have step := Nat.mul_le_mul_right m (Nat.succ_le_of_lt hi)
  simp only [Nat.succ_mul] at step
  omega

theorem active_addresses (c : Call) (x : Input α) (valid : Admissible c x)
    (j : Job) (hj : Legal c j) (k d : Nat) (hk : k < x.counts j.batch) (hd : d < c.dimensions) :
    pageOffset c x j k < c.batch*c.maximum ∧
    logicAddress c x j k < c.heads*c.tokens ∧
    valueAddress c x j k d < c.pages*c.heads*c.dimensions ∧
    address c j d < c.batch*c.heads*c.dimensions := by
  have range := valid.lengths j.batch hj.1
  refine ⟨?_,?_,?_,?_⟩
  · simpa [pageOffset,Nat.add_assoc] using flat_lt j.batch (c.maximum-x.counts j.batch+k) c.batch c.maximum hj.1 (by omega)
  · simpa [logicAddress,Nat.add_assoc] using flat_lt j.head (x.starts j.batch+k) c.heads c.tokens hj.2 (by omega)
  · exact flat_lt _ _ _ _ (flat_lt _ _ _ _ (valid.pages j hj k hk) hj.2) hd
  · exact flat_lt _ _ _ _ (flat_lt _ _ _ _ hj.1 hj.2) hd

/-- The active source memory accesses stay within their actual allocations;
all offsets fit signed int32 without changing the kernel's integer arithmetic. -/
theorem active_int32 (c : Call) (x : Input α) (valid : Admissible c x)
    (j : Job) (hj : Legal c j) (k d : Nat) (hk : k < x.counts j.batch) (hd : d < c.dimensions) :
    pageOffset c x j k < 2^31 ∧ logicAddress c x j k < 2^31 ∧
    valueAddress c x j k d < 2^31 ∧ address c j d < 2^31 := by
  have bounds := active_addresses c x valid j hj k d hk hd
  have := valid.locations_width;have := valid.logits_width;have := valid.values_width;have := valid.output_width
  omega

/-- Every reached loop increment and every formed masked logit/page lane fits
signed int32, including the final 63 inactive lanes. No divisibility assumption. -/
theorem padded_int32 (c : Call) (x : Input α) (valid : Admissible c x)
    (j : Job) (hj : Legal c j) (cur lane : Nat) (reached : cur < x.counts j.batch) (hlane : lane < 64) :
    cur+64 < 2^31 ∧ cur+lane < 2^31 ∧
    pageOffset c x j (cur+lane) < 2^31 ∧ logicAddress c x j (cur+lane) < 2^31 := by
  have range := valid.lengths j.batch hj.1
  have pagebase := Nat.mul_le_mul_right c.maximum (Nat.succ_le_of_lt hj.1)
  have logicbase := Nat.mul_le_mul_right c.tokens (Nat.succ_le_of_lt hj.2)
  simp only [Nat.succ_mul] at pagebase logicbase
  have := valid.maximum_width;have := valid.locations_width;have := valid.logits_width
  unfold pageOffset logicAddress
  omega

#axiomsClean active_addresses
#axiomsClean active_int32
#axiomsClean padded_int32
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked
