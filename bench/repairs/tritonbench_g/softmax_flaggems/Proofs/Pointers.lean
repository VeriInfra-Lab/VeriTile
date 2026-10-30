import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Integer
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Pointers
open VeriTile.Bench.Repairs
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
set_option maxHeartbeats 3000000

/-- Either the forward heuristic or any of the six backward trials. -/
def Selected (waves : WavesAboveOne) (d : Dimensions) (sms : Nat) (c : LaunchConfig) : Prop :=
  c=forwardConfig waves d sms ∨ ∃ choice, c=backwardConfig d choice

theorem config_bounds (waves : WavesAboveOne) (d : Dimensions) (sms : Nat) (c : LaunchConfig)
    (selected : Selected waves d sms c) : 0<c.B ∧ 0<c.C ∧ c.B<2^17 ∧ c.C<2^17 := by
  rcases selected with h|⟨choice,h⟩ <;> subst c
  · exact ⟨(forwardConfig_positive waves d sms).1,(forwardConfig_positive waves d sms).2,
      Integer.forward_config_bounds waves d sms⟩
  · exact ⟨(backwardConfig_positive d choice).1,(backwardConfig_positive d choice).2,
      Integer.backward_config_bounds d choice⟩

/-- All tile lanes and the carried value immediately after the final loop. -/
theorem tile_bounds (D B p lane : Nat) (hd : 0<D) (hb : 0<B)
    (hp : p<RealAddress.rounds D B) (hl : lane<B) :
    p*B<D ∧ p*B+lane<D+B ∧ (p+1)*B+lane≤D+2*B := by
  have base := (RealAddress.tile_start_lt_iff hd hb).mpr hp
  simp only [Nat.add_mul,Nat.one_mul]
  omega

theorem widened_tile (D B p lane : Nat) (hd : 0<D) (hb : 0<B)
    (db : D<2^31) (bb : B<2^17) (hp : p<RealAddress.rounds D B) (hl : lane<B) :
    Ksoftmax.Native.widened32 p=(p:Int) ∧ Ksoftmax.Native.widened32 lane=(lane:Int) ∧
    Integer.add (Integer.mul p B) lane=(p*B+lane : Nat) := by
  have b := tile_bounds D B p lane hd hb hp hl
  have pp : p≤p*B := by nlinarith
  have p1 := Integer.nat64 (p*B) (by omega)
  have p2 := Integer.nat64 (p*B+lane) (by omega)
  simp only [Nat.cast_mul,Nat.cast_add] at p1 p2
  exact ⟨Ksoftmax.Native.widened32_exact p (by omega),Ksoftmax.Native.widened32_exact lane (by omega),
    by simp [Integer.add,Integer.mul,Ksoftmax.Native.add,Ksoftmax.Native.mul,p1,p2]⟩

/-- Actual selected non-inner launches: every padded and carried source offset
is exact; each active load/store is in the contiguous Tensor extent. -/
theorem noninner (x : TensorMetadata) (contract : TensorContract x) (ax : Fin x.shape.length)
    (live : x.numel≠0) (waves : WavesAboveOne) (sms : Nat) (c : LaunchConfig)
    (selected : Selected waves (dimensions x.shape ax) sms c)
    (m n k : Nat) (hm : m<(dimensions x.shape ax).M)
    (hn : n≤(dimensions x.shape ax).N+2*c.B) (hk : k<(dimensions x.shape ax).K+c.C) :
    let d := dimensions x.shape ax
    Integer.noninner m n k d.N d.K=(m*d.N*d.K+n*d.K+k : Nat) ∧
    (n<d.N → k<d.K → m*d.N*d.K+n*d.K+k<x.numel) := by
  obtain ⟨hm0,hn0,hk0,mb,nb,kb,total⟩ := Integer.dimension_bounds x contract ax live
  have cb := config_bounds waves (dimensions x.shape ax) sms c selected
  refine ⟨Integer.noninner_exact _ _ _ _ _ _ _ _ hm hn0 hk0 total cb.2.2.1 cb.2.2.2 hn hk,?_⟩
  intro an ak
  have count := (FPHost.dimensions_positive x.shape ax live).2.2.2
  change m*(dimensions x.shape ax).N*(dimensions x.shape ax).K+n*(dimensions x.shape ax).K+k<x.shape.prod
  rw [count]
  exact RealAddress.non_inner_bound hm an ak

/-- Inner launches include padded groups of rows in backward configurations. -/
theorem inner (x : TensorMetadata) (contract : TensorContract x) (ax : Fin x.shape.length)
    (live : x.numel≠0) (waves : WavesAboveOne) (sms : Nat) (c : LaunchConfig)
    (selected : Selected waves (dimensions x.shape ax) sms c) (innerCase : (dimensions x.shape ax).K=1)
    (m n : Nat) (hm : m<(dimensions x.shape ax).M+c.C) (hn : n≤(dimensions x.shape ax).N+2*c.B) :
    let d := dimensions x.shape ax
    Integer.inner m n d.N=(m*d.N+n : Nat) ∧
    (m<d.M → n<d.N → m*d.N+n<x.numel) := by
  obtain ⟨hm0,hn0,hk0,mb,nb,kb,total⟩ := Integer.dimension_bounds x contract ax live
  have cb := config_bounds waves (dimensions x.shape ax) sms c selected
  have count := (FPHost.dimensions_positive x.shape ax live).2.2.2
  simp only [innerCase,Nat.mul_one] at total count
  refine ⟨Integer.inner_exact _ _ _ _ _ _ hm0 hn0 total cb.2.2.1 cb.2.2.2 hm hn,?_⟩
  intro am an
  have b := RealAddress.non_inner_bound (K:=1) am an (by decide : 0<1)
  simpa only [Nat.mul_one,Nat.add_zero,←count] using b

/-- Span validity is an ordinary Tensor/allocation binding; input alignment is
exactly the checked nonempty guard. Masked-off pointers need no dereference. -/
theorem active_pointer (x : TensorMetadata) (contract : TensorContract x)
    (span : x.dataPtr.toNat+4*x.numel≤2^64) (j : Nat) (hj : j<x.numel) :
    (Ksoftmax.Native.pointer x.dataPtr.toNat j).toNat=x.dataPtr.toNat+4*j ∧
    x.dataPtr.toNat≤(Ksoftmax.Native.pointer x.dataPtr.toNat j).toNat ∧
    (Ksoftmax.Native.pointer x.dataPtr.toNat j).toNat+4≤x.dataPtr.toNat+4*x.numel ∧
    (Ksoftmax.Native.pointer x.dataPtr.toNat j).toNat%4=0 := by
  have a := Ksoftmax.Native.active_pointer x.dataPtr.toNat (4*x.numel) x.numel j (by omega) span hj
  have al := contract.2.2.2.2.2.2 (by omega)
  exact ⟨a.1,a.2.1,a.2.2,by rw [a.1];omega⟩

#axiomsClean config_bounds
#axiomsClean widened_tile
#axiomsClean noninner
#axiomsClean inner
#axiomsClean active_pointer
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Pointers
