import bench.tritonbench_g.lightning_attention.Proofs.HostMetadata
import bench.tritonbench_g.chunk_linear_attn.Physical
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton
set_option maxHeartbeats 6000000

/-- The integer element offset used by every tensor pointer in the three
kernels. Unlike an access bound, formed offsets include inactive mask lanes. -/
def formedOffset (d : Dimensions) (width head time feature : Nat) : Nat :=
  head*d.N*width+time*width+feature

theorem channel_count_exact (d : Dimensions) (h : Contract d) : d.NV*d.BM=d.E := by
  obtain ⟨k,hk⟩ := bm_divides d h
  have hp := bm_positive d h
  have hk' : k*d.BM=d.E := by nlinarith [hk]
  have he : (d.E+d.BM-1)/d.BM=k := by
    apply Nat.div_eq_of_lt_le
    · rw [hk']
      omega
    · rw [Nat.add_mul,Nat.one_mul,hk']
      omega
  change ((d.E+d.BM-1)/d.BM)*d.BM=d.E
  rw [he,hk']

theorem channel_coordinate (d : Dimensions) (h : Contract d) (channel lane : Nat)
    (hc : channel<d.NV) (hl : lane<d.BM) : channel*d.BM+lane<d.E := by
  have hm := Nat.mul_le_mul_right d.BM (Nat.succ_le_of_lt hc)
  have he := channel_count_exact d h
  simp only [Nat.succ_mul] at hm
  omega

/-- One extra 64-lane vector after the final forward iteration is included;
the source's reverse initialization uses only the first 32 of these lanes. -/
theorem formed_row (d : Dimensions) (block lane : Nat) (hb : block≤d.NT) (hl : lane<64) :
    block*64+lane<d.N+128 := by
  have hp := time_padding d
  have hm := Nat.mul_le_mul_right 64 hb
  omega

theorem forward_sub_row (d : Dimensions) (block sub lane : Nat)
    (hb : block<d.NT) (hs : sub<2) (hl : lane<32) :
    (block*2+sub)*32+lane<d.N+128 := by
  have hp := time_padding d
  omega

theorem forward_advance_row (d : Dimensions) (step lane : Nat)
    (hs : step≤d.NT*2) (hl : lane<32) : step*32+lane<d.N+128 := by
  have hp := time_padding d
  omega

theorem reverse_sub_nonnegative (d : Dimensions) (block sub : Nat)
    (hb : block<d.NT) (hs : sub<2) : (block*2+sub+1)*32≤d.NT*64 := by omega
theorem reverse_sub_row (d : Dimensions) (block sub lane : Nat)
    (hb : block<d.NT) (hs : sub<2) (hl : lane<32) :
    d.NT*64-(block*2+sub+1)*32+lane<d.N+128 := by
  have hp := time_padding d
  omega
theorem reverse_initial_row (d : Dimensions) (lane : Nat) (hl : lane<32) :
    d.NT*64+lane<d.N+128 := formed_row d d.NT lane le_rfl (by omega)

/-- The source guard has room for all padded lanes and all pointer advances,
including reverse pointers initially formed beyond the final logical row. -/
theorem formed_offset (d : Dimensions) (width head time feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (ht : time<d.N+128) (hf : feature<width) :
    formedOffset d width head time feature<d.formedLimit := by
  have ha := Nat.mul_le_mul_right d.N (Nat.succ_le_of_lt hh)
  have hb := Nat.mul_le_mul_right width (show head*d.N+time+1≤d.B*d.H*d.N+128 by
    simp only [Nat.succ_mul] at ha
    omega)
  have hc := Nat.mul_le_mul_left (d.B*d.H*d.N+128) hw
  unfold formedOffset Dimensions.formedLimit
  nlinarith

theorem formed_machine (d : Dimensions) (h : Contract d) (width head time feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (ht : time<d.N+128) (hf : feature<width) :
    formedOffset d width head time feature<2^31 :=
  lt_trans (formed_offset d width head time feature hw hh ht hf) h.machine

theorem forward_query_formed (d : Dimensions) (head block lane feature : Nat)
    (hh : head<d.B*d.H) (hb : block≤d.NT) (hl : lane<64) (hf : feature<d.D) :
    formedOffset d d.D head (block*64+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ (le_max_left _ _) hh (formed_row d block lane hb hl) hf
theorem forward_value_formed (d : Dimensions) (h : Contract d)
    (head block lane channel feature : Nat) (hh : head<d.B*d.H)
    (hb : block≤d.NT) (hl : lane<64) (hc : channel<d.NV) (hf : feature<d.BM) :
    formedOffset d d.E head (block*64+lane) (channel*d.BM+feature)<d.formedLimit :=
  formed_offset d _ _ _ _ (le_max_right _ _) hh (formed_row d block lane hb hl)
    (channel_coordinate d h channel feature hc hf)

theorem intra_formed (d : Dimensions) (width head block lane feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (hb : block<d.NT) (hl : lane<64) (hf : feature<width) :
    formedOffset d width head (block*64+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ hw hh (formed_row d block lane (Nat.le_of_lt hb) hl) hf
theorem inter_forward_formed (d : Dimensions) (width head block sub lane feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (hb : block<d.NT) (hs : sub<2) (hl : lane<32) (hf : feature<width) :
    formedOffset d width head ((block*2+sub)*32+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ hw hh (forward_sub_row d block sub lane hb hs hl) hf
theorem inter_advance_formed (d : Dimensions) (width head step lane feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (hs : step≤d.NT*2) (hl : lane<32) (hf : feature<width) :
    formedOffset d width head (step*32+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ hw hh (forward_advance_row d step lane hs hl) hf
theorem reverse_initial_formed (d : Dimensions) (width head lane feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H) (hl : lane<32) (hf : feature<width) :
    formedOffset d width head (d.NT*64+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ hw hh (reverse_initial_row d lane hl) hf
theorem inter_reverse_formed (d : Dimensions) (width head block sub lane feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (hb : block<d.NT) (hs : sub<2) (hl : lane<32) (hf : feature<width) :
    formedOffset d width head (d.NT*64-(block*2+sub+1)*32+lane) feature<d.formedLimit :=
  formed_offset d _ _ _ _ hw hh (reverse_sub_row d block sub lane hb hs hl) hf

/-- Every nonnegative multiplication/addition intermediate in a formed
pointer is bounded by its complete element offset. -/
theorem formed_intermediates (d : Dimensions) (h : Contract d) (width head time feature : Nat)
    (hw : width≤max d.D d.E) (hh : head<d.B*d.H)
    (ht : time<d.N+128) (hf : feature<width) :
    head*d.N<2^31 ∧ head*d.N*width<2^31 ∧ time*width<2^31 ∧
      head*d.N*width+time*width<2^31 ∧ feature<2^31 := by
  have ho := formed_machine d h width head time feature hw hh ht hf
  have hp : 1≤width := by omega
  have hm := Nat.mul_le_mul_left (head*d.N) hp
  simp only [Nat.mul_one] at hm
  unfold formedOffset at ho
  omega

theorem dimension_machine (d : Dimensions) (h : Contract d) :
    d.B<2^31 ∧ d.H<2^31 ∧ d.N<2^31 ∧ d.D<2^31 ∧ d.E<2^31 ∧
      d.NT*64+63<2^31 := by
  have hp := h.machine
  have hh := heads_bound d h
  have hb := Nat.mul_le_mul_left d.B h.h_pos
  have he := Nat.mul_le_mul_right d.H h.b_pos
  have hn := Nat.mul_le_mul_right d.N hh.1
  have hm : 1≤max d.D d.E := by have := h.d_min; have := le_max_left d.D d.E; omega
  have hg := Nat.mul_le_mul_left (d.B*d.H*d.N+128) hm
  have hc := Nat.mul_le_mul_right (max d.D d.E) (show 1≤d.B*d.H*d.N+128 by omega)
  have hd := le_max_left d.D d.E
  have hv := le_max_right d.D d.E
  have ht := time_padding d
  simp only [Nat.mul_one,Nat.one_mul] at hb he hn hg hc
  unfold Dimensions.formedLimit at hp
  omega

/-- Reverse subtraction is ordinary signed subtraction with a nonnegative
result; this identity connects it to the Nat row model without truncation. -/
theorem reverse_signed_row (d : Dimensions) (block sub lane : Nat)
    (hb : block<d.NT) (hs : sub<2) :
    ((d.NT*64 : Nat) : Int)-(((block*2+sub+1)*32 : Nat) : Int)+(lane : Int)=
      ((d.NT*64-(block*2+sub+1)*32+lane : Nat) : Int) := by
  have hn := reverse_sub_nonnegative d block sub hb hs
  omega

theorem reverse_pointer_step (d : Dimensions) (width head step lane feature : Nat)
    (hs : step<d.NT*2) :
    formedOffset d width head (d.NT*64-step*32+lane) feature-32*width=
      formedOffset d width head (d.NT*64-(step+1)*32+lane) feature := by
  have hr : 32≤d.NT*64-step*32 := by omega
  have he : d.NT*64-step*32+lane=(d.NT*64-(step+1)*32+lane)+32 := by omega
  unfold formedOffset
  rw [he]
  simp only [Nat.add_mul]
  omega

/-- Signed causal-mask differences are bounded independently of tensor size. -/
theorem mask_index_machine (left right : Nat) (hl : left<64) (hr : right<64) :
    -(2^31 : Int)≤(left : Int)-(right : Int) ∧
      (left : Int)-(right : Int)<2^31 := by omega

abbrev PhysicalBuffer := ChunkLinearAttn.Physical.Buffer
/-- Ordinary ABI room for the exact bases supplied by the runtime, measured
in FP32 words. Formed pointer arithmetic is translated to 64-bit byte addresses. -/
structure PointerABI (d : Dimensions) (buffers : List PhysicalBuffer) : Prop where
  room : ∀ b∈buffers, b.base+d.formedLimit<2^61
theorem pointer_safe (d : Dimensions) (buffers : List PhysicalBuffer) (abi : PointerABI d buffers)
    (b : PhysicalBuffer) (hb : b∈buffers) (offset : Nat) (ho : offset<d.formedLimit) :
    4*(b.base+offset)<2^63 := by
  have hr := abi.room b hb
  norm_num at hr ⊢
  omega

#axiomsClean channel_count_exact
#axiomsClean formed_offset
#axiomsClean forward_query_formed
#axiomsClean forward_value_formed
#axiomsClean intra_formed
#axiomsClean inter_forward_formed
#axiomsClean inter_advance_formed
#axiomsClean reverse_initial_formed
#axiomsClean inter_reverse_formed
#axiomsClean formed_intermediates
#axiomsClean dimension_machine
#axiomsClean reverse_signed_row
#axiomsClean reverse_pointer_step
#axiomsClean mask_index_machine
#axiomsClean pointer_safe
#specNonCircular PointerABI avoiding [fwdMemory,fwdOutput,fwdCarry]
end VeriTile.Bench.Repairs.LightningAttention.Host
