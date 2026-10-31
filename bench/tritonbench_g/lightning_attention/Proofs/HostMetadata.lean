import bench.tritonbench_g.lightning_attention.Kernels
import bench.tritonbench_g.lightning_attention.Proofs.Model
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
set_option maxHeartbeats 4000000
structure Dimensions where
  (B H N D E : Nat)
def Dimensions.NT (d : Dimensions) := (d.N+63)/64
def Dimensions.BM (d : Dimensions) := min d.E 32
def Dimensions.NV (d : Dimensions) := (d.E+d.BM-1)/d.BM
def Dimensions.parameters (d : Dimensions) : Parameters :=
  ⟨d.B,d.H,d.N,d.D,d.E,64,d.BM,d.NT,32,2⟩
def Dimensions.formedLimit (d : Dimensions) := (d.B*d.H*d.N+128)*max d.D d.E
structure Contract (d : Dimensions) : Prop where
  b_pos : 0<d.B
  h_pos : 0<d.H
  n_pos : 0<d.N
  d_min : 16≤d.D
  e_min : 16≤d.E
  d_power : ∃ k, d.D=2^k
  e_power : ∃ k, d.E=2^k
  time_axis : d.NT≤65535
  channel_axis : d.NV≤65535
  machine : d.formedLimit<2^31

theorem bm_legal (d : Dimensions) (h : Contract d) : d.BM=16 ∨ d.BM=32 := by
  have hemin := h.e_min
  obtain ⟨k,hk⟩ := h.e_power
  by_cases he : d.E<32
  · have hk5 : k≤4 := by
      by_contra hn
      have hp := Nat.pow_le_pow_right (by decide : 0<2) (show 5≤k by omega)
      norm_num at hp
      omega
    interval_cases k <;> norm_num at hk
    all_goals first | omega | simp_all [Dimensions.BM]
  · right
    exact Nat.min_eq_right (by omega)

theorem bm_positive (d : Dimensions) (h : Contract d) : 0<d.BM := by
  rcases bm_legal d h with he|he <;> omega
theorem bm_divides (d : Dimensions) (h : Contract d) : d.BM∣d.E := by
  by_cases he : d.E≤32
  · simp [Dimensions.BM,Nat.min_eq_left he]
  · obtain ⟨k,hk⟩ := h.e_power
    have hk5 : 5≤k := by
      by_contra hn
      have hk4 : k≤4 := by omega
      interval_cases k <;> norm_num at hk <;> omega
    rw [Dimensions.BM,Nat.min_eq_right (by omega),hk]
    exact Nat.pow_dvd_pow 2 hk5
theorem time_cover (d : Dimensions) : d.N≤d.NT*64 := by
  have ht := Nat.mod_lt (d.N+63) (by decide : 0<64)
  have hd := Nat.mod_add_div (d.N+63) 64
  unfold Dimensions.NT
  omega
theorem time_padding (d : Dimensions) : d.NT*64≤d.N+63 := by
  unfold Dimensions.NT
  exact Nat.div_mul_le_self _ _
theorem time_positive (d : Dimensions) (h : Contract d) : 0<d.NT := by
  have hc := time_cover d
  have hp := h.n_pos
  omega
theorem channel_positive (d : Dimensions) (h : Contract d) : 0<d.NV := by
  have he := h.e_min
  have hb := bm_positive d h
  unfold Dimensions.NV
  exact Nat.div_pos (by omega) hb
theorem heads_bound (d : Dimensions) (h : Contract d) : 0<d.B*d.H ∧ d.B*d.H<2^31 := by
  refine ⟨Nat.mul_pos h.b_pos h.h_pos,?_⟩
  have hn : 1≤d.N := h.n_pos
  have hw : 1≤max d.D d.E := by have := h.d_min; have := le_max_left d.D d.E; omega
  have hhead : d.B*d.H≤d.B*d.H*d.N := by simpa using Nat.mul_le_mul_left (d.B*d.H) hn
  have hsize : d.B*d.H*d.N+128≤d.formedLimit := by
    simpa [Dimensions.formedLimit] using Nat.mul_le_mul_left (d.B*d.H*d.N+128) hw
  have hm := h.machine
  omega

inductive Stage where | forward | intra | inter
def grid (d : Dimensions) : Stage → Nat × Nat × Nat
  | .forward => (d.B*d.H,d.NV,1)
  | .intra => (d.B*d.H,d.NT,1)
  | .inter => (d.B*d.H,1,1)
theorem launch_axes (d : Dimensions) (h : Contract d) (stage : Stage) :
    0<(grid d stage).1 ∧ (grid d stage).1<2^31 ∧
      0<(grid d stage).2.1 ∧ (grid d stage).2.1≤65535 ∧ (grid d stage).2.2=1 := by
  have hh := heads_bound d h
  cases stage with
  | forward => exact ⟨hh.1,hh.2,channel_positive d h,h.channel_axis,rfl⟩
  | intra => exact ⟨hh.1,hh.2,time_positive d h,h.time_axis,rfl⟩
  | inter => exact ⟨hh.1,hh.2,by simp [grid],by simp [grid],rfl⟩

#axiomsClean bm_legal
#axiomsClean bm_divides
#axiomsClean launch_axes
#specNonCircular Contract avoiding [KernelOnly.forward,KernelOnly.intra,KernelOnly.inter]
end VeriTile.Bench.Repairs.LightningAttention.Host
