import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.NextPower
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPHostCore
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Native
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Integer
set_option maxHeartbeats 3000000
open VeriTile.Bench.Repairs
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid

abbrev signed64 := VarLenCopy.signed64
abbrev mul := Ksoftmax.Native.mul
abbrev add := Ksoftmax.Native.add

def noninner (m n k N K : Nat) : Int := add (add (mul (mul m N) K) (mul n K)) k
def inner (m n N : Nat) : Int := add (mul m N) n

theorem factor_bounds (M N K : Nat) (hm : 0<M) (hn : 0<N) (hk : 0<K) :
    M≤M*N*K ∧ N≤M*N*K ∧ K≤M*N*K ∧ N*K≤M*N*K := by
  have m : M≤M*N := by nlinarith
  have n : N≤M*N := by nlinarith
  have mn : M*N≤M*N*K := by nlinarith
  have k : K≤M*N*K := by have h := Nat.mul_pos hm hn;nlinarith
  have nk : N*K≤M*N*K := by nlinarith
  exact ⟨by omega,by omega,k,nk⟩

theorem dimension_bounds (x : TensorMetadata) (h : TensorContract x)
    (ax : Fin x.shape.length) (live : x.numel≠0) :
    let d := dimensions x.shape ax
    0<d.M ∧ 0<d.N ∧ 0<d.K ∧ d.M<2^31 ∧ d.N<2^31 ∧ d.K<2^31 ∧ d.M*d.N*d.K<2^31 := by
  obtain ⟨hm,hn,hk,hcount⟩ := FPHost.dimensions_positive x.shape ax live
  have total : (dimensions x.shape ax).M*(dimensions x.shape ax).N*(dimensions x.shape ax).K<2^31 := by
    rw [←hcount];exact h.2.2.2.2.2.1
  have hb := factor_bounds _ _ _ hm hn hk
  exact ⟨hm,hn,hk,by omega,by omega,by omega,total⟩

theorem nat64 (n : Nat) (hn : n<2^63) : signed64 (n : Int)=(n : Int) :=
  VarLenCopy.signed64_exact _ (by omega) (by exact_mod_cast hn)

/-- Includes padded arange lanes and the post-loop carried offset. All tile
extents are below the ordinary signed-i32 arange primitive limit. -/
theorem noninner_exact (M N K B C m n k : Nat)
    (hm : m<M) (hn : 0<N) (hk : 0<K) (total : M*N*K<2^31)
    (hB : B<2^17) (hC : C<2^17) (pn : n≤N+2*B) (pk : k<K+C) :
    noninner m n k N K=(m*N*K+n*K+k : Nat) := by
  have factors := factor_bounds M N K (by omega) hn hk
  have kb : K<2^31 := by omega
  have mb : m*N*K<M*N*K := Nat.mul_lt_mul_of_pos_right (Nat.mul_lt_mul_of_pos_right hm hn) hk
  have mb' : m*N≤m*N*K := by
    simpa using Nat.mul_le_mul_left (m*N) (Nat.succ_le_of_lt hk)
  have bk : B*K<2^48 := by
    have h1 := Nat.mul_lt_mul_of_pos_right hB hk
    have h2 := Nat.mul_lt_mul_of_pos_left kb (by decide : 0<(2^17 : Nat))
    norm_num at h2
    omega
  have nk : n*K≤N*K+2*(B*K) := by
    simpa [Nat.add_mul,Nat.mul_assoc] using Nat.mul_le_mul_right K pn
  have last : m*N*K+n*K+k<2^63 := by omega
  have p1 := nat64 (m*N) (by omega)
  have p2 := nat64 (m*N*K) (by omega)
  have p3 := nat64 (n*K) (by omega)
  have p4 := nat64 (m*N*K+n*K) (by omega)
  have p5 := nat64 (m*N*K+n*K+k) last
  simp only [Nat.cast_mul,Nat.cast_add] at p1 p2 p3 p4 p5
  simp only [noninner,add,mul,Ksoftmax.Native.add,Ksoftmax.Native.mul,p1,p2,p3,p4,p5,Nat.cast_mul,Nat.cast_add]

theorem inner_exact (M N B C m n : Nat) (hm : 0<M) (hn : 0<N)
    (total : M*N<2^31) (hB : B<2^17) (hC : C<2^17) (pm : m<M+C) (pn : n≤N+2*B) :
    inner m n N=(m*N+n : Nat) := by
  have nb : N<2^31 := by
    have h : N≤M*N := by nlinarith
    omega
  have cn : C*N<2^48 := by
    have h1 := Nat.mul_lt_mul_of_pos_right hC hn
    have h2 := Nat.mul_lt_mul_of_pos_left nb (by decide : 0<(2^17 : Nat))
    norm_num at h2
    omega
  have mm : m*N<M*N+C*N := by
    simpa [Nat.add_mul] using Nat.mul_lt_mul_of_pos_right pm hn
  have last : m*N+n<2^63 := by omega
  have p1 := nat64 (m*N) (by omega)
  have p2 := nat64 (m*N+n) last
  simp only [Nat.cast_mul,Nat.cast_add] at p1 p2
  simp only [inner,add,mul,Ksoftmax.Native.add,Ksoftmax.Native.mul,p1,p2,Nat.cast_mul,Nat.cast_add]

theorem reverse_exact (N B tile : Nat) (hn : 0<N) (hb : 0<B) (nb : N<2^31)
    (visit : tile*B<N) :
    signed64 ((RealAddress.previous N B : Int)-(tile*B : Nat))=
      (RealAddress.previous N B-tile*B : Nat) := by
  have count := (RealAddress.tile_start_lt_iff hn hb).mp visit
  have le : tile*B≤RealAddress.previous N B := by
    rw [RealAddress.previous_eq]
    exact Nat.mul_le_mul_right B (by omega)
  rw [←Int.natCast_sub le]
  exact nat64 _ (by have := RealAddress.previous_lt hn hb;omega)

theorem forward_config_bounds (waves : WavesAboveOne) (d : Dimensions) (sms : Nat) :
    (forwardConfig waves d sms).B<2^17 ∧ (forwardConfig waves d sms).C<2^17 := by
  unfold forwardConfig
  split_ifs
  · have cp := tileKLoop_positive waves d.M d.K sms 1 (by decide)
    have cb := tileKLoop_bound waves d.M d.K sms 1 (by decide) (by decide)
    have div := Nat.div_le_self (8192+heurTileK waves d.M d.K sms-1) (heurTileK waves d.M d.K sms)
    change RealAddress.rounds 8192 (heurTileK waves d.M d.K sms)<2^17 ∧ heurTileK waves d.M d.K sms<2^17
    unfold heurTileK at div ⊢
    unfold RealAddress.rounds
    omega
  · change heurTileNInner d.N<2^17 ∧ (1:Nat)<2^17
    constructor
    · unfold heurTileNInner
      split_ifs with h
      · exact next_power_bound d.N h
      · decide
    · decide

theorem backward_config_bounds (d : Dimensions) (choice : Fin 6) :
    (backwardConfig d choice).B<2^17 ∧ (backwardConfig d choice).C<2^17 := by
  have hb : autotuneTile choice≤1024 := by
    fin_cases choice <;> decide
  have div := Nat.div_le_self 1024 (autotuneTile choice)
  unfold backwardConfig
  split_ifs <;> dsimp only <;> omega

#axiomsClean dimension_bounds
#axiomsClean noninner_exact
#axiomsClean inner_exact
#axiomsClean reverse_exact
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Integer
