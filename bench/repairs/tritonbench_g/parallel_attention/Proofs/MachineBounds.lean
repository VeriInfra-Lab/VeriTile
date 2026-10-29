import bench.repairs.tritonbench_g.parallel_attention.Proofs.Contract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Grid
namespace VeriTile.Bench.Repairs.ParallelAttention.Machine
open VeriTile.Bench.Repairs.ParallelAttention
set_option maxHeartbeats 8000000

def heads (s : Shape) := s.B*s.H
def valueShards (s : Shape) := blocks s.V s.BV
def shardHeads (s : Shape) := valueShards s*heads s
def largestWidth (s : Shape) := max s.K s.V
/-- The Python guard includes all scratch shards, padded rows/features, and
the pointer state reached after a final tl.advance. -/
def formedLimit (s : Shape) := (valueShards s*s.B*s.H*s.T+256)*largestWidth s+128
def Empty (s : Shape) : Prop := s.B=0 ∨ s.H=0 ∨ s.T=0
def Active (s : Shape) : Prop := 0 < s.B ∧ 0 < s.H ∧ 0 < s.T
structure Guard (s : Shape) : Prop where
  heads_axis : heads s ≤ 65535
  time_axis : blocks s.T 64 ≤ 65535
  formed : formedLimit s < 2^31
/-- Empty inputs return allocated empty tensors without a kernel launch. -/
def Admission (s : Shape) : Prop := Empty s ∨ Guard s
def axisGrid (s : Shape) (L : Nat) :=
  (blocks s.K s.BK*valueShards s,blocks s.T L,heads s)
def launches (s : Shape) (L : Nat) : List (Nat × Nat × Nat) :=
  if s.B=0 ∨ s.H=0 ∨ s.T=0 then [] else [axisGrid s L]
def Nonwrap32 (n : Nat) : Prop := n < 2^31
def formedOffset (s : Shape) (width packedHead time feature : Nat) :=
  packedHead*s.T*width+time*width+feature

theorem empty_no_launch (s : Shape) (L : Nat) (h : Empty s) : launches s L=[] := by
  unfold launches
  exact if_pos h
theorem nonempty_active (s : Shape) (h : ¬Empty s) : Active s := by
  unfold Empty at h
  unfold Active
  omega
theorem admitted_active (s : Shape) (h : Admission s) (a : Active s) : Guard s := by
  rcases h with h|h
  · unfold Empty at h
    rcases a with ⟨hb,hh,ht⟩
    omega
  · exact h
theorem empty_cells (s : Shape) (h : Empty s) (kind : OutputKind) : kind.cells s=0 := by
  rcases h with h|h|h <;> simp [OutputKind.cells,h]

theorem selected_cap (n : Nat) : selectedTile n ≤ 128 := by
  unfold selectedTile
  split_ifs <;> omega
theorem selected_ge (n : Nat) : 16 ≤ selectedTile n := by
  unfold selectedTile
  split_ifs <;> omega
theorem tile_caps (s : Shape) (h : ShapeContract s) :
    16 ≤ s.BK ∧ s.BK ≤ 128 ∧ 16 ≤ s.BV ∧ s.BV ≤ 128 := by
  rw [h.key_tile,h.value_tile]
  exact ⟨selected_ge _,selected_cap _,selected_ge _,selected_cap _⟩
theorem key_shards (s : Shape) (h : ShapeContract s) : blocks s.K s.BK=1 :=
  one_key_shard s.K s.BK h.key_positive (h.key_fits s)
theorem shards_positive (s : Shape) (h : ShapeContract s) : 0 < valueShards s := by
  have hc := blocks_cover s.V s.BV (h.value_tile_positive s)
  have hv := h.value_positive
  unfold valueShards
  by_contra hn
  have hz : blocks s.V s.BV=0 := by omega
  rw [hz] at hc
  omega

theorem block_start (n width block : Nat) (hw : 0 < width) (hb : block < blocks n width) :
    block*width < n := by
  have hm := Nat.mul_le_mul_right width (Nat.succ_le_of_lt hb)
  have hd := Nat.div_mul_le_self (n+width-1) width
  unfold blocks at hm
  simp only [Nat.succ_mul] at hm
  omega
theorem blocks_padding (n width : Nat) : blocks n width*width ≤ n+width-1 :=
  Nat.div_mul_le_self _ _
theorem padded_feature (s : Shape) (h : ShapeContract s) (shard lane : Nat)
    (hs : shard < valueShards s) (hl : lane < s.BV) : shard*s.BV+lane < s.V+128 := by
  have ht := block_start s.V s.BV shard (h.value_tile_positive s) hs
  have hc := (tile_caps s h).2.2.2
  omega
theorem padded_key_feature (s : Shape) (h : ShapeContract s) (shard lane : Nat)
    (hs : shard < blocks s.K s.BK) (hl : lane < s.BK) : shard*s.BK+lane < s.K+128 := by
  have cap := tile_caps s h
  have ht := block_start s.K s.BK shard (by omega) hs
  omega
theorem padded_row (s : Shape) (L chunk lane : Nat) (hL : L=64 ∨ L=128)
    (hc : chunk < blocks s.T L) (hl : lane < L) : chunk*L+lane < s.T+128 := by
  have hp : 0 < L := by omega
  have ht := block_start s.T L chunk hp hc
  omega
theorem small_row (s : Shape) (block lane : Nat) (hb : block < blocks s.T 32) (hl : lane < 32) :
    block*32+lane < s.T+32 := by
  have ht := block_start s.T 32 block (by decide) hb
  omega
theorem advanced_row (s : Shape) (L chunk lane : Nat) (hL : L=64 ∨ L=128)
    (hc : chunk < blocks s.T L) (hl : lane < 32) : (chunk+1)*L+lane < s.T+256 := by
  have ht := block_start s.T L chunk (by omega) hc
  simp only [Nat.add_mul,Nat.one_mul]
  omega
theorem prefix_row (s : Shape) (L chunk step lane : Nat) (hL : L=64 ∨ L=128)
    (hc : chunk < blocks s.T L) (hs : step*32 ≤ chunk*L) (hl : lane < 32) :
    step*32+lane < s.T+256 := by
  have ht := block_start s.T L chunk (by omega) hc
  omega
theorem diagonal_row (s : Shape) (L chunk step lane : Nat) (hL : L=64 ∨ L=128)
    (hc : chunk < blocks s.T L) (hs : step*32 ≤ L) (hl : lane < 32) :
    chunk*L+step*32+lane < s.T+256 := by
  have ht := block_start s.T L chunk (by omega) hc
  omega

/-- A single arithmetic envelope covers direct inputs, O/Z/DV shards, and
the larger NV-sharded DQ/DK allocations. Feature masks do not affect it. -/
theorem formed_offset (s : Shape) (width packedHead time feature : Nat)
    (hw : width ≤ largestWidth s) (hh : packedHead < shardHeads s)
    (ht : time < s.T+256) (hf : feature < width+128) :
    formedOffset s width packedHead time feature < formedLimit s := by
  have ha := Nat.mul_le_mul_right s.T (Nat.succ_le_of_lt hh)
  have hb := Nat.mul_le_mul_right width (show packedHead*s.T+time+1 ≤ shardHeads s*s.T+256 by
    simp only [Nat.succ_mul] at ha
    omega)
  have hc := Nat.mul_le_mul_left (shardHeads s*s.T+256) hw
  have he : shardHeads s*s.T=valueShards s*s.B*s.H*s.T := by unfold shardHeads heads; ring
  unfold formedOffset formedLimit
  rw [he] at hb hc
  nlinarith

theorem formed_machine (s : Shape) (g : Guard s) (width packedHead time feature : Nat)
    (hw : width ≤ largestWidth s) (hh : packedHead < shardHeads s)
    (ht : time < s.T+256) (hf : feature < width+128) :
    Nonwrap32 (formedOffset s width packedHead time feature) :=
  lt_trans (formed_offset s width packedHead time feature hw hh ht hf) g.formed

theorem packed_head (s : Shape) (shard head : Nat) (hs : shard < valueShards s) (hh : head < heads s) :
    head+heads s*shard < shardHeads s := by
  unfold shardHeads
  nlinarith
theorem direct_head (s : Shape) (h : ShapeContract s) (head : Nat) (hh : head < heads s) :
    head < shardHeads s := by
  have hn := shards_positive s h
  have hm := Nat.mul_le_mul_right (heads s) hn
  unfold shardHeads
  omega

theorem query_formed (s : Shape) (h : ShapeContract s) (head time key lane : Nat)
    (hh : head < heads s) (ht : time < s.T+256) (hk : key < blocks s.K s.BK) (hl : lane < s.BK) :
    formedOffset s s.K head time (key*s.BK+lane) < formedLimit s :=
  formed_offset s _ _ _ _ (le_max_left _ _) (direct_head s h head hh) ht
    (padded_key_feature s h key lane hk hl)
theorem value_formed (s : Shape) (h : ShapeContract s) (head time shard lane : Nat)
    (hh : head < heads s) (ht : time < s.T+256) (hs : shard < valueShards s) (hl : lane < s.BV) :
    formedOffset s s.V head time (shard*s.BV+lane) < formedLimit s :=
  formed_offset s _ _ _ _ (le_max_right _ _) (direct_head s h head hh) ht
    (padded_feature s h shard lane hs hl)
theorem query_scratch_formed (s : Shape) (h : ShapeContract s) (head time shard key lane : Nat)
    (hh : head < heads s) (ht : time < s.T+256) (hs : shard < valueShards s)
    (hk : key < blocks s.K s.BK) (hl : lane < s.BK) :
    formedOffset s s.K (head+heads s*shard) time (key*s.BK+lane) < formedLimit s :=
  formed_offset s _ _ _ _ (le_max_left _ _) (packed_head s shard head hs hh) ht
    (padded_key_feature s h key lane hk hl)
theorem value_scratch_formed (s : Shape) (h : ShapeContract s) (head time shard key lane : Nat)
    (hh : head < heads s) (ht : time < s.T+256) (hs : shard < valueShards s)
    (hk : key < blocks s.K s.BK) (hl : lane < s.BV) :
    formedOffset s s.V (head+heads s*key) time (shard*s.BV+lane) < formedLimit s := by
  have hz : key=0 := by rw [key_shards s h] at hk; omega
  simpa [hz] using value_formed s h head time shard lane hh ht hs hl
theorem normalizer_formed (s : Shape) (h : ShapeContract s) (head time key : Nat)
    (hh : head < heads s) (ht : time < s.T+256) (hk : key < blocks s.K s.BK) :
    (head+heads s*key)*s.T+time < formedLimit s := by
  have hz : key=0 := by rw [key_shards s h] at hk; omega
  have hw : 1 ≤ largestWidth s := le_trans h.key_positive (le_max_left _ _)
  have hb := formed_offset s 1 head time 0 hw (direct_head s h head hh) ht (by decide)
  simpa [formedOffset,hz] using hb

theorem largest_positive (s : Shape) (h : ShapeContract s) : 1 ≤ largestWidth s :=
  le_trans h.key_positive (le_max_left _ _)
theorem scalar_ranges (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s) :
    Nonwrap32 (valueShards s) ∧ Nonwrap32 (heads s) ∧ Nonwrap32 (shardHeads s) ∧
      Nonwrap32 s.T ∧ Nonwrap32 s.K ∧ Nonwrap32 s.V ∧ Nonwrap32 (s.T*s.K) ∧
      Nonwrap32 (s.T*s.V) ∧ Nonwrap32 (s.T+255) := by
  have hh : 1 ≤ heads s := Nat.mul_pos a.1 a.2.1
  have hn := shards_positive s h
  have hsh := Nat.mul_le_mul_left (valueShards s) hh
  have hheads := Nat.mul_le_mul_right (heads s) hn
  have hprod := Nat.mul_le_mul_right s.T (Nat.mul_pos hn hh)
  have hrows := Nat.mul_le_mul_left (shardHeads s) a.2.2
  have hw := largest_positive s h
  have hbase := Nat.mul_le_mul_left (shardHeads s*s.T+256) hw
  have hwidth := Nat.mul_le_mul_right (largestWidth s) (show 1 ≤ shardHeads s*s.T+256 by omega)
  have hTK := Nat.mul_le_mul s.T.le_refl (le_max_left s.K s.V)
  have hTV := Nat.mul_le_mul s.T.le_refl (le_max_right s.K s.V)
  have hT := Nat.mul_le_mul_right (largestWidth s) (show s.T ≤ shardHeads s*s.T+256 by
    unfold shardHeads
    omega)
  have hg := g.formed
  have hk := le_max_left s.K s.V
  have hv := le_max_right s.K s.V
  have he : valueShards s*s.B*s.H*s.T=shardHeads s*s.T := by unfold shardHeads heads; ring
  unfold formedLimit at hg
  rw [he] at hg
  unfold Nonwrap32 largestWidth at *
  simp only [Nat.mul_one,Nat.one_mul] at *
  unfold shardHeads at *
  omega

theorem launch_axes (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s)
    (L : Nat) (hL : L=64 ∨ L=128) :
    0 < (axisGrid s L).1 ∧ (axisGrid s L).1 < 2^31 ∧
    0 < (axisGrid s L).2.1 ∧ (axisGrid s L).2.1 ≤ 65535 ∧
    0 < (axisGrid s L).2.2 ∧ (axisGrid s L).2.2 ≤ 65535 := by
  have hp := shards_positive s h
  have hr := (scalar_ranges s h g a).1
  have ht := blocks_cover s.T L (by omega)
  have hc : 0 < blocks s.T L := by
    have hn := a.2.2
    by_contra hz
    have he : blocks s.T L=0 := by omega
    rw [he] at ht
    omega
  have htime : blocks s.T L ≤ 65535 := by
    rcases hL with rfl|rfl
    · exact g.time_axis
    · have hg := g.time_axis
      unfold blocks at hg ⊢
      omega
  simp only [axisGrid,key_shards s h,Nat.one_mul]
  exact ⟨hp,hr,hc,htime,Nat.mul_pos a.1 a.2.1,g.heads_axis⟩

theorem reverse_loop_start (s : Shape) (a : Active s) :
    32 ≤ blocks s.T 32*32 ∧ blocks s.T 32*32-32 < s.T := by
  have hc := blocks_cover s.T 32 (by decide)
  have hp := blocks_padding s.T 32
  have hn := a.2.2
  have hz : 0 < blocks s.T 32 := by
    by_contra hzero
    have he : blocks s.T 32=0 := by omega
    rw [he] at hc
    omega
  omega
theorem reverse_stop_nonnegative (chunk : Nat) : 32 ≤ (chunk+1)*64 := by omega
theorem reverse_control_machine (s : Shape) (h : ShapeContract s) (g : Guard s)
    (a : Active s) (i : Int) (hi : -32 ≤ i ∧ i < (s.T : Int)+128) :
    -(2^31 : Int) ≤ i ∧ i < 2^31 := by
  have hr := (scalar_ranges s h g a).2.2.2.2.2.2.2.2
  unfold Nonwrap32 at hr
  omega

/-- Tensor block pointers add canonical source strides. -/
theorem source_address (s : Shape) (width head time feature : Nat) :
    head*(s.T*width)+time*width+feature=formedOffset s width head time feature := by
  unfold formedOffset
  ring
theorem scratch_address (s : Shape) (width shard head time feature : Nat) :
    (head+s.B*s.H*shard)*(s.T*width)+time*width+feature=
      shard*(s.B*s.H*s.T*width)+dense s.T width head time feature := by
  unfold dense
  ring

/-- A runtime base has room for every formed FP32-word address. No allocation
must contain inactive lanes; only active accesses require allocated cells. -/
structure PointerABI (s : Shape) (bases : List Nat) : Prop where
  room : ∀ base∈bases, base+formedLimit s < 2^61
theorem pointer_safe (s : Shape) (bases : List Nat) (abi : PointerABI s bases)
    (base : Nat) (hb : base∈bases) (offset : Nat) (ho : offset < formedLimit s) :
    4*(base+offset) < 2^63 := by
  have hp := abi.room base hb
  norm_num at hp ⊢
  omega

#axiomsClean empty_no_launch
#axiomsClean empty_cells
#axiomsClean padded_feature
#axiomsClean advanced_row
#axiomsClean formed_offset
#axiomsClean query_formed
#axiomsClean value_formed
#axiomsClean query_scratch_formed
#axiomsClean value_scratch_formed
#axiomsClean normalizer_formed
#axiomsClean scalar_ranges
#axiomsClean launch_axes
#axiomsClean reverse_loop_start
#axiomsClean reverse_control_machine
#axiomsClean source_address
#axiomsClean scratch_address
#axiomsClean pointer_safe
#specNonCircular Guard avoiding [combinedRun,hostRun]
#specNonCircular Admission avoiding [combinedRun,hostRun]
#specNonCircular PointerABI avoiding [combinedRun,hostRun]
end VeriTile.Bench.Repairs.ParallelAttention.Machine
