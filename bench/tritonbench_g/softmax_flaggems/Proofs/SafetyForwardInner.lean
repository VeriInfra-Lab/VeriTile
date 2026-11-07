import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyAligned

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem first_bound (N B cur : Nat) (hN : 0<N) (hB : 0<B)
    (aligned : cur%B=0) (hc : cur<previous N B) (i : Fin B) : cur+i.val<N := by
  have hd := Nat.dvd_of_mod_eq_zero aligned
  obtain ⟨q,hq⟩ := hd
  have hp : previous N B=(RealAddress.rounds N B-1)*B := RealAddress.previous_eq N B
  rw [hq,Nat.mul_comm B q,hp] at hc
  have hq' : q<RealAddress.rounds N B-1 := Nat.lt_of_mul_lt_mul_right hc
  have hh := Nat.mul_le_mul_right B hq'
  have last : previous N B<N := RealAddress.previous_lt hN hB
  rw [hq,Nat.mul_comm B q]
  rw [hp] at last
  have hi := i.isLt
  simp only [Nat.succ_mul] at hh
  omega

theorem reverse_bound (N B cur : Nat) (hN : 0<N) (hB : 0<B)
    (hBN : B<N) (hc : B≤cur) (i : Fin B) : previous N B-cur+i.val<N := by
  have hp : previous N B<N := RealAddress.previous_lt hN hB
  have hi := i.isLt
  omega

theorem accumulate_safe (A : Algebra α) (base s : State α) (X Y : RegionName)
    (masked : Bool) (row rows N B cur : Nat) (v : Acc α B)
    (hs : Ready base X Y row N B v s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N → k<bounds X)
    (activeBound : ∀ i : Fin B, active masked N cur (i,PUnit.unit) → cur+i.val<N) :
    traceSafeList A bounds (rangeBody (code Y X rows N B) (if masked then 9 else 8))
      (s.setReg "start_n" .int [] (fun _ => (cur : Int))) := by
  rcases hs with ⟨hxp,hyp,hprev,hm,hz,hmem⟩
  cases masked <;>
    simp [rangeBody,code,branch,Source.forward_inner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner_kernel,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
      ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
      TileShape.axisDim,State.setReg,and_assoc,hxp,hyp,hprev,hm,hz,hmem]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  · intro i
    exact hx _ (HostGrid.inner_bound hrow (activeBound i (by simp [active])))
  · intro i hi
    exact hx _ (HostGrid.inner_bound hrow hi)

theorem emit_safe (A : Algebra α) (base s : State α) (X Y : RegionName)
    (masked : Bool) (row rows N B cur : Nat) (mx z : α)
    (hcur : cur≤previous N B)
    (hs : WriteReady base X Y row N B mx z s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N → k<bounds X) (hy : ∀ k,k<rows*N → k<bounds Y)
    (activeBound : ∀ i : Fin B, active masked N (previous N B-cur) (i,PUnit.unit) → previous N B-cur+i.val<N) :
    traceSafeList A bounds (rangeBody (code Y X rows N B) (if masked then 17 else 18))
      (if masked then s.setReg "start_n" .nat [] (fun _ => cur)
        else s.setReg "start_n" .int [] (fun _ => (cur : Int))) := by
  have subCast : (previous N B : Int)-(cur : Int) = ((previous N B-cur : Nat) : Int) :=
    (Int.ofNat_sub hcur).symm
  rcases hs with ⟨hxp,hyp,hprev,hm,hz,hmem⟩
  cases masked <;>
    simp [rangeBody,code,branch,Source.forward_inner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner_kernel,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
      ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
      TileShape.axisDim,State.setReg,and_assoc,hxp,hyp,hprev,hm,hz,hmem]
  all_goals try simp only [subCast,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  · constructor
    · intro i
      exact hx _ (HostGrid.inner_bound hrow (activeBound i (by simp [active])))
    · intro i
      exact hy _ (HostGrid.inner_bound hrow (activeBound i (by simp [active])))
  · constructor
    · intro i hi
      exact hx _ (HostGrid.inner_bound hrow hi)
    · intro i hi
      exact hy _ (HostGrid.inner_bound hrow hi)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
