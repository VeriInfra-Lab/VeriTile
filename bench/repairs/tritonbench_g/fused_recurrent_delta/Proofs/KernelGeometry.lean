import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelContract
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardResults
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardFrame
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelGeometry
open VeriTile.Triton KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxHeartbeats 10000000

def Writes (kind : Kind) (s : Shape) (f : Flags) (p : Program kind s) : RegionName → Nat → Prop :=
  match kind with
  | .forward => KernelForwardResults.Writes s f p.head p.value
  | .backward => KernelBackwardFrame.Writes s f p.head p.value

theorem region_injective : Function.Injective region := by
  intro a b eq
  cases a <;> cases b <;> simp_all [region]

theorem key_offset (s : Shape) (f : Flags) (r : Region)
    (role : r=.q ∨ r=.k ∨ r=.dq ∨ r=.dk) (bh t i : Nat) :
    keyAddress s bh t i=Geometry.offset s f ⟨r,bh,t,i⟩ := by
  rcases role with rfl|rfl|rfl|rfl <;>
    simp only [keyAddress,Geometry.offset,Geometry.rows,Geometry.cols] <;> ring
theorem value_offset (s : Shape) (f : Flags) (r : Region)
    (role : r=.v ∨ r=.o ∨ r=.do_ ∨ r=.dv ∨ (r=.beta ∧ f.vectorBeta=Bool.true) ∨
      (r=.dbeta ∧ f.vectorBeta=Bool.true)) (bh iv BV t j : Nat) :
    valAddress s bh iv BV t j=Geometry.offset s f ⟨r,bh,t,iv*BV+j⟩ := by
  rcases role with rfl|rfl|rfl|rfl|⟨rfl,h⟩|⟨rfl,h⟩ <;>
    simp_all [valAddress,Geometry.offset,Geometry.rows,Geometry.cols] <;> ring
theorem state_offset (s : Shape) (f : Flags) (r : Region)
    (role : r=.h0 ∨ r=.ht ∨ r=.dht ∨ r=.dh0) (bh iv BV i j : Nat) :
    stateAddress s bh iv BV i j=Geometry.offset s f ⟨r,bh,i,iv*BV+j⟩ := by
  rcases role with rfl|rfl|rfl|rfl <;>
    simp only [stateAddress,Geometry.offset,Geometry.rows,Geometry.cols] <;> ring
theorem scalar_offset (s : Shape) (f : Flags) (r : Region)
    (role : r=.beta ∨ r=.dbeta) (scalar : f.vectorBeta=Bool.false) (bh t : Nat) :
    bh*s.T+t=Geometry.offset s f ⟨r,bh,t,0⟩ := by
  rcases role with rfl|rfl <;> simp [Geometry.offset,Geometry.rows,Geometry.cols,scalar]

theorem scratch_plane (s : Shape) (p : Program .backward s) : plane s p.head p.value<s.BNV*s.BH := by
  have h := Geometry.mixed_bound s.BNV s.BH p.value p.head p.valueInside p.headInside
  simpa only [plane,Shape.BH,Nat.mul_assoc,Nat.add_comm] using h

theorem key_bound (s : Shape) (P bh t i : Nat) (hp : bh<P) (ht : t<s.T) (hi : i<s.K) :
    keyAddress s bh t i<P*s.T*s.K := by
  have h := Geometry.mixed_bound (P*s.T) s.K (bh*s.T+t) i
    (Geometry.mixed_bound P s.T bh t hp ht) hi
  convert h using 1 <;> simp only [keyAddress] <;> ring
theorem value_bound (s : Shape) (P bh iv BV t j : Nat) (hp : bh<P) (ht : t<s.T)
    (hj : iv*BV+j<s.V) : valAddress s bh iv BV t j<P*s.T*s.V := by
  have h := Geometry.mixed_bound (P*s.T) s.V (bh*s.T+t) (iv*BV+j)
    (Geometry.mixed_bound P s.T bh t hp ht) hj
  convert h using 1 <;> simp only [valAddress] <;> ring
theorem state_bound (s : Shape) (bh iv BV i j : Nat) (hp : bh<s.BH) (hi : i<s.K)
    (hj : iv*BV+j<s.V) : stateAddress s bh iv BV i j<s.BH*s.K*s.V := by
  have h := Geometry.mixed_bound (s.BH*s.K) s.V (bh*s.K+i) (iv*BV+j)
    (Geometry.mixed_bound s.BH s.K bh i hp hi) hj
  convert h using 1 <;> simp only [stateAddress] <;> ring

/-- Every source store is a cell of the exact raw kernel argument shape.
In particular, vector dbeta uses the program's own scratch plane and column. -/
theorem output_bounds (kind : Kind) (s : Shape) (f : Flags) (p : Program kind s)
    (name : RegionName) (off : Nat) (active : Writes kind s f p name off) :
    ∃ r,r∈KernelContract.outputRoles kind f ∧ region r=name ∧ off<Geometry.capacity s f r := by
  cases kind with
  | forward =>
    rcases active with ⟨role,t,j,ht,hj,rfl⟩|⟨enabled,rfl,i,j,hi,hj,rfl⟩
    · have bound := value_bound s s.BH p.head p.value s.FBV t j.val p.headInside ht hj
      rcases role with rfl|rfl
      · exact ⟨.v,by simp [KernelContract.outputRoles],rfl,bound⟩
      · exact ⟨.o,by simp [KernelContract.outputRoles],rfl,bound⟩
    · exact ⟨.ht,by simp [KernelContract.outputRoles,enabled],rfl,
        state_bound s p.head p.value s.FBV i.val j.val p.headInside hi hj⟩
  | backward =>
    rcases active with ⟨role,t,i,ht,hi,rfl⟩|⟨rfl,t,j,ht,hj,rfl⟩|⟨rfl,beta⟩|⟨enabled,rfl,i,j,hi,hj,rfl⟩
    · have bound := key_bound s (s.BNV*s.BH) (plane s p.head p.value) t i.val (scratch_plane s p) ht hi
      rcases role with rfl|rfl
      · exact ⟨.dk,by simp [KernelContract.outputRoles],rfl,bound⟩
      · exact ⟨.dq,by simp [KernelContract.outputRoles],rfl,bound⟩
    · exact ⟨.dv,by simp [KernelContract.outputRoles],rfl,
        value_bound s s.BH p.head p.value s.BBV t j.val p.headInside ht hj⟩
    · refine ⟨.dbeta,by simp [KernelContract.outputRoles],rfl,?_⟩
      cases hb : f.vectorBeta <;> simp only [hb,ite_false,ite_true,Bool.false_eq_true] at beta
      · obtain ⟨t,ht,rfl⟩ := beta
        simpa [Geometry.capacity,Geometry.planes,Geometry.rows,Geometry.cols,hb] using
          Geometry.mixed_bound (s.BNV*s.BH) s.T (plane s p.head p.value) t (scratch_plane s p) ht
      · obtain ⟨t,j,ht,hj,rfl⟩ := beta
        simpa [Geometry.capacity,Geometry.planes,Geometry.rows,Geometry.cols,hb] using
          value_bound s (s.BNV*s.BH) (plane s p.head p.value) p.value s.BBV t j.val (scratch_plane s p) ht hj
    · exact ⟨.dh0,by simp [KernelContract.outputRoles,enabled],rfl,
        state_bound s p.head p.value s.BBV i.val j.val p.headInside hi hj⟩

theorem outputs_are_arguments (kind : Kind) (f : Flags) (r : Region)
    (hr : r∈KernelContract.outputRoles kind f) : r∈KernelContract.roles kind f := by
  cases kind <;> cases hi : f.initial <;> cases hf : f.final <;>
    cases hg : f.initialGrad <;> cases ht : f.terminalGrad <;>
    simp_all [KernelContract.outputRoles,KernelContract.roles] <;> tauto

theorem active_i32 (kind : Kind) (s : Shape) (f : Flags) (safe : KernelContract.Domain s f)
    (p : Program kind s) (name : RegionName) (off : Nat) (active : Writes kind s f p name off) :
    (BitVec.ofInt 32 off).toInt=off := by
  obtain ⟨r,_,_,bound⟩ := output_bounds kind s f p name off active
  have cap := Index.capacity_safe s f safe.toSafe r
  exact Index.i32_exact _ (by omega) (by omega)

theorem value_owner (kind : Kind) (s : Shape) (bh v : Nat) (hh : bh<s.BH) (hv : v<s.V) :
    ∃ p : Program kind s,∃ j : Fin (blockV kind s),p.head=bh ∧ p.value*blockV kind s+j.val=v := by
  have hp : 0<blockV kind s := by
    cases kind <;> simp [blockV]; exact (Geometry.tile_pos s).2.1; exact (Geometry.tile_pos s).2.2
  obtain ⟨inside,lane,eq⟩ := Geometry.owner_exists s.V (blockV kind s) v hp hv
  have grid : ceil s.V (blockV kind s)=gridV kind s := by cases kind <;> rfl
  exact ⟨⟨bh,v/blockV kind s,hh,by simpa [grid] using inside⟩,⟨v%blockV kind s,lane⟩,rfl,eq⟩

#axiomsClean output_bounds
#axiomsClean active_i32
#axiomsClean value_owner
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelGeometry
