import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeIndex
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.HostPlan
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open NativeIndex
noncomputable section
set_option maxHeartbeats 1500000

/-- Actual synchronous host read of normalized integer metadata. The empty
batch branch returns zero; max/item neither mutates metadata nor rounds it. -/
def metadataMaximum (c : Call) (r : Regions) (s : BlockState) : Nat :=
  (Finset.univ : Finset (Fin c.batches)).sup (fun b => ((inputs s r).sequence b.val).toNat)

def maximum (c : Call) : Nat :=
  (Finset.univ : Finset (Fin c.batches)).sup (fun b => c.length b.val)

theorem metadataMaximum_eq {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) : metadataMaximum c r s = maximum c := by
  unfold metadataMaximum maximum
  congr 1
  funext b
  rw [hm.length b b.isLt, Int.toNat_natCast]

theorem length_le_maximum (c : Call) (b : Nat) (hb : b < c.batches) :
    c.length b ≤ maximum c :=
  Finset.le_sup (f := fun x : Fin c.batches => c.length x.val) (Finset.mem_univ ⟨b,hb⟩)

theorem maximum_lt_int32 {c : Call} (hs : Shape c) : maximum c < 2^31 := by
  have hmax : maximum c ≤ 2147483647 := by
    apply Finset.sup_le
    intro b _
    have := hs.int32Length b b.isLt
    norm_num at this
    omega
  norm_num
  omega

def effective (c : Call) (r : Regions) (s : BlockState) : Call :=
  capCall c (metadataMaximum c r s)

theorem effective_memory {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) : Memory (effective c r s) r A s := by
  cases hm with
  | mk shape start length startExtent lengthExtent qBounded kBounded vBounded oBounded
      outQuery outKey outValue outStart outSequence =>
    exact ⟨cap_shape _ shape, start, length, startExtent, lengthExtent,
      qBounded, kBounded, vBounded, oBounded, outQuery, outKey, outValue, outStart, outSequence⟩

theorem effective_covered {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (b t : Nat) (hb : b < c.batches) (ht : t < c.length b) :
    t < (effective c r s).blocks*128 ↔ t < c.blocks*128 := by
  change t < ((capMaxInputLen c.maxInputLen (metadataMaximum c r s)+127)/128)*128 ↔ _
  rw [metadataMaximum_eq hm]
  exact cap_preserves_active c.maxInputLen (maximum c) t
    (lt_of_lt_of_le ht (length_le_maximum c b hb))

theorem effective_complete {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (hc : c.completeGrid) : (effective c r s).completeGrid := by
  apply cap_complete
  · intro b hb
    rw [metadataMaximum_eq hm]
    exact length_le_maximum c b hb
  · exact hc

/-- A Python range(0,n,limit) base, represented without narrowing n. -/
abbrev Base (n limit : Nat) := { b : Fin n // b.val % limit = 0 }
/-- One actual local program coordinate inside its clipped native chunk. -/
abbrev Axis (n limit : Nat) := (b : Base n limit) × Fin (min limit (n-b.val.val))

def global {n limit : Nat} (j : Axis n limit) : Fin n :=
  ⟨j.1.val.val+j.2.val, by have := (lt_min_iff.mp j.2.isLt).2; have := j.1.val.isLt; omega⟩

def split {n limit : Nat} (hl : 0 < limit) (i : Fin n) : Axis n limit :=
  ⟨⟨⟨(i.val/limit)*limit, (axis_coverage hl i.isLt).1.1⟩, by simp⟩,
    ⟨i.val%limit, (axis_coverage hl i.isLt).1.2⟩⟩

@[simp] theorem global_split {n limit : Nat} (hl : 0 < limit) (i : Fin n) :
    global (split hl i) = i := Fin.ext (axis_coverage hl i.isLt).2

theorem global_injective {n limit : Nat} (hl : 0 < limit) :
    Function.Injective (@global n limit) := by
  intro x y he
  have hx := x.1.property
  have hy := y.1.property
  have hxl := (lt_min_iff.mp x.2.isLt).1
  have hyl := (lt_min_iff.mp y.2.isLt).1
  have hg : x.1.val.val+x.2.val = y.1.val.val+y.2.val := congrArg Fin.val he
  have hbase : x.1.val.val = y.1.val.val := by
    have hm := congrArg (fun a => a % limit) hg
    simp only [Nat.add_mod, hx, hy, Nat.zero_add, Nat.mod_eq_of_lt hxl,
      Nat.mod_eq_of_lt hyl] at hm
    omega
  have hb : x.1 = y.1 := Subtype.ext (Fin.ext hbase)
  cases x with | mk xb xl =>
    cases y with | mk yb yl =>
      dsimp at hb hg
      cases hb
      congr
      apply Fin.ext
      omega

@[simp] theorem split_global {n limit : Nat} (hl : 0 < limit) (j : Axis n limit) :
    split hl (global j) = j := global_injective hl (global_split hl _)

def axisEquiv (n limit : Nat) (hl : 0 < limit) : Axis n limit ≃ Fin n where
  toFun := global
  invFun := split hl
  left_inv := split_global hl
  right_inv := global_split hl

abbrev Job (c : Call) := Axis c.batches 2147483647 × Axis c.q.heads 65535 × Axis c.blocks 65535

def index (c : Call) (j : Job c) : GridIndex c.grid :=
  c.index (global j.1).val (global j.2.1).val (global j.2.2).val
    (global j.1).isLt (global j.2.1).isLt (global j.2.2).isLt

def splitIndex (c : Call) (i : GridIndex c.grid) : Job c :=
  (split (by decide) ⟨(c.launch i).batch, launch_batch_lt c i⟩,
   split (by decide) ⟨(c.launch i).head, launch_head_lt c i⟩,
   split (by decide) ⟨(c.launch i).block, launch_block_lt c i⟩)

@[simp] theorem index_splitIndex (c : Call) (i : GridIndex c.grid) : index c (splitIndex c i) = i := by
  apply launch_injective
  simp only [index, splitIndex, global_split, index_launch]

@[simp] theorem splitIndex_index (c : Call) (j : Job c) : splitIndex c (index c j) = j := by
  simp only [splitIndex, index, index_launch]
  exact Prod.ext (split_global (by decide) j.1)
    (Prod.ext (split_global (by decide) j.2.1) (split_global (by decide) j.2.2))

def jobEquiv (c : Call) : Job c ≃ GridIndex c.grid where
  toFun := index c
  invFun := splitIndex c
  left_inv := splitIndex_index c
  right_inv := index_splitIndex c

/-- Native grid sizes of the actual three clipped chunks. -/
def nativeGrid (c : Call) (j : Job c) : Grid :=
  ⟨[min 2147483647 (c.batches-j.1.1.val.val),
    min 65535 (c.q.heads-j.2.1.1.val.val), min 65535 (c.blocks-j.2.2.1.val.val)]⟩

theorem nativeGrid_limits (c : Call) (j : Job c) :
    (nativeGrid c j).dim 0 ≤ 2147483647 ∧
    (nativeGrid c j).dim 1 ≤ 65535 ∧ (nativeGrid c j).dim 2 ≤ 65535 := by
  change min _ _ ≤ _ ∧ min _ _ ≤ _ ∧ min _ _ ≤ _
  exact ⟨Nat.min_le_left _ _,Nat.min_le_left _ _,Nat.min_le_left _ _⟩

/-- Lexicographic source nesting: batch base, head base, block base, followed
by one allowed local GPU program order. All digits are bounded by radix. -/
def orderKey (c : Call) (j : Job c) : Nat :=
  let radix := c.batches+c.q.heads+c.blocks+2147483648
  (((((j.1.1.val.val*radix+j.2.1.1.val.val)*radix+j.2.2.1.val.val)*radix+
    j.1.2.val)*radix+j.2.1.2.val)*radix+j.2.2.2.val)

def jobs (c : Call) : List (Job c) :=
  (Finset.univ : Finset (Job c)).toList.mergeSort (fun a b => orderKey c a ≤ orderKey c b)

theorem jobs_complete (c : Call) : (jobs c).Nodup ∧ ∀ j : Job c, j ∈ jobs c := by
  have hp := List.mergeSort_perm ((Finset.univ : Finset (Job c)).toList)
    (fun a b => decide (orderKey c a ≤ orderKey c b))
  constructor
  · exact hp.nodup_iff.mpr (Finset.nodup_toList _)
  · intro j
    apply hp.mem_iff.mpr
    exact Finset.mem_toList.mpr (Finset.mem_univ j)

theorem mapped_jobs_complete (c : Call) :
    ((jobs c).map (index c)).Nodup ∧
    ∀ i : GridIndex c.grid, i ∈ (jobs c).map (index c) := by
  constructor
  · exact List.Nodup.map (jobEquiv c).injective (jobs_complete c).1
  · intro i
    exact List.mem_map.mpr ⟨splitIndex c i,(jobs_complete c).2 _,index_splitIndex c i⟩

end
end VeriTile.Bench.Repairs.ContextAttnNopad.HostPlan
