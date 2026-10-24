import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativePrefix

/-! A nonempty arithmetic domain at the two actually recorded native compile
profiles. Numeric frontend predicates are proved; they are not a Lean model
of CUDA compilation, launch resource limits, compiler hints or GPU execution. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeConfig
open NativeDomain
set_option maxHeartbeats 3000000

def blockK : TypedSource.Storage → Nat
  | .fp16 => 64
  | .f8e4 => 128

def stages : TypedSource.Storage → Nat
  | .fp16 => 3
  | .f8e4 => 4

/-- Only the exact compile specializations in the pinned sm90 receipt.
Runtime dimensions/strides are i32; SMS=132 is not generalized to all devices. -/
def RecordedCompileProfile (p : Parameters) (storage : TypedSource.Storage)
    (warps pipelineStages : Nat) : Prop :=
  p.BM = 128 ∧ p.BN = 256 ∧ p.BK = blockK storage ∧ p.GM = 8 ∧ p.SMS = 132 ∧
    warps = 8 ∧ pipelineStages = stages storage

def ArangeChecks (n : Nat) : Prop :=
  0 < n ∧ n < 2^32 ∧ (∃ k : Nat, n = 2^k)

/-- Numeric checks read from installed Triton 3.7.1: power-of-two block
dimensions, 2^20 tensor-element cap and sm90 dot minimum K of 16/32. -/
def FrontendChecks (p : Parameters) (storage : TypedSource.Storage) (warps : Nat) : Prop :=
  ArangeChecks p.BM ∧ ArangeChecks p.BN ∧ ArangeChecks p.BK ∧
    p.BM*p.BK ≤ 1048576 ∧ p.BK*p.BN ≤ 1048576 ∧ p.BM*p.BN ≤ 1048576 ∧
    1 ≤ p.BM ∧ 1 ≤ p.BN ∧ (match storage with | .fp16 => 16 | .f8e4 => 32) ≤ p.BK ∧
    0 < warps ∧ (∃ k : Nat, warps = 2^k)

theorem recorded_frontend (p : Parameters) (storage : TypedSource.Storage)
    (warps pipelineStages : Nat) (h : RecordedCompileProfile p storage warps pipelineStages) :
    FrontendChecks p storage warps := by
  obtain ⟨hm,hn,hk,_hg,_hs,hw,_hst⟩ := h
  unfold FrontendChecks
  rw [hm,hn,hk,hw]
  have h128 : ArangeChecks 128 := ⟨by decide,by decide,⟨7,by decide⟩⟩
  have h256 : ArangeChecks 256 := ⟨by decide,by decide,⟨8,by decide⟩⟩
  have h64 : ArangeChecks 64 := ⟨by decide,by decide,⟨6,by decide⟩⟩
  cases storage
  · exact ⟨h128,h256,h64,by decide,by decide,by decide,by decide,by decide,by decide,by decide,⟨3,by decide⟩⟩
  · exact ⟨h128,h256,h128,by decide,by decide,by decide,by decide,by decide,by decide,by decide,⟨3,by decide⟩⟩

/-- Full native tile sizes with 17x9 output tiles, a shortened final group,
two tiles assigned to PID0, and a partial final K block in both profiles. -/
def witness (storage : TypedSource.Storage) : Parameters where
  A := "A"
  B := "B"
  C := "C"
  M := 2176
  N := 2304
  K := 129
  sam := 129
  sak := 1
  sbk := 2304
  sbn := 1
  scm := 2304
  scn := 1
  BM := 128
  BN := 256
  BK := blockK storage
  GM := 8
  SMS := 132

theorem witness_profile (storage : TypedSource.Storage) :
    RecordedCompileProfile (witness storage) storage 8 (stages storage) := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem witness_admissible (storage : TypedSource.Storage) : Admissible (witness storage) := by
  cases storage <;> constructor <;>
    norm_num [witness,blockK,limit,rows,columns,count,total,rowPad,columnPad,kPad,RealMath.blocks]

theorem witness_frontend (storage : TypedSource.Storage) :
    FrontendChecks (witness storage) storage 8 :=
  recorded_frontend _ _ _ _ (witness_profile storage)

/-- The exact wrapper grid is positive and at most 132 for these recorded
profiles. Eight 32-thread warps give 256 threads, within the numeric block cap;
resource availability and host/device agreement remain external obligations. -/
theorem recorded_launch_bounds (p : Parameters) (storage : TypedSource.Storage)
    (warps pipelineStages : Nat) (h : RecordedCompileProfile p storage warps pipelineStages)
    (hd : Admissible p) :
    0 < grid p ∧ grid p ≤ 132 ∧ warps*32 = 256 ∧ warps*32 ≤ 1024 := by
  obtain ⟨_hm,_hn,_hk,_hg,hs,hw,_hst⟩ := h
  have ht : 0 < total p := Nat.mul_pos (rows_pos p hd) (columns_pos p hd)
  refine ⟨lt_min hd.sms ht,?_,?_,?_⟩
  · simpa only [grid,hs] using (Nat.min_le_left p.SMS (total p))
  · omega
  · omega

theorem witness_nonempty (storage : TypedSource.Storage) :
    grid (witness storage) = 132 ∧
    work (witness storage) 0 = 2 ∧
    Schedule.groupSize (rows (witness storage)) (columns (witness storage)) 8 152 = 1 ∧
    (witness storage).K % (witness storage).BK ≠ 0 := by
  cases storage <;> norm_num [grid,work,witness,blockK,rows,columns,total,
    RealMath.blocks,Schedule.tilesFor,Schedule.groupSize,Schedule.groupBase]

/-- This witness's M/N coordinate tiles are unclamped contiguous ranges with
aligned starts. Interpretation of the compiler hint metadata stays separate. -/
theorem witness_coordinates (storage : TypedSource.Storage) (r c rl cl : Nat)
    (hr : r < rows (witness storage)) (hc : c < columns (witness storage))
    (hrl : rl < 128) (hcl : cl < 256) :
    r*128+rl < (witness storage).M ∧ c*256+cl < (witness storage).N ∧
    r*128%128 = 0 ∧ c*256%256 = 0 := by
  have hr' : r < 17 := by norm_num [rows,witness,RealMath.blocks] at hr; exact hr
  have hc' : c < 9 := by norm_num [columns,witness,RealMath.blocks] at hc; exact hc
  change r*128+rl < 2176 ∧ c*256+cl < 2304 ∧ r*128%128=0 ∧ c*256%256=0
  exact ⟨by omega,by omega,by omega,by omega⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeConfig
