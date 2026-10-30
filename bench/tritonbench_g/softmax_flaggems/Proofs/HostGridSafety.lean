import bench.tritonbench_g.softmax_flaggems.HostCorrect

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000

/-- Safety follows the actual intermediate states of sequential `run`. -/
noncomputable def RunSafe (k : Kernel) (bounds : RegionBounds) (gx gy : Nat) :
    List Program → BlockState → Prop
  | [],_ => True
  | p::ps,s => k.TraceSafe bounds (setup gx gy p s) ∧
      ∀ t, exec k (setup gx gy p s)=some t → RunSafe k bounds gx gy ps t

theorem runSafe_of_each (k : Kernel) (bounds : RegionBounds) (gx gy : Nat)
    (ps : List Program)
    (h : ∀ p ∈ ps, ∀ s, k.TraceSafe bounds (setup gx gy p s)) (s : BlockState) :
    RunSafe k bounds gx gy ps s := by
  induction ps generalizing s with
  | nil => trivial
  | cons p ps ih =>
      exact ⟨h p (by simp) s,fun t _ ↦ ih (fun p hp ↦ h p (by simp [hp])) t⟩

theorem forward_inner_grid_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (one : Bool)
    (h : RealForwardInner.ExecutionDomain one X Y N B)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) (ps : List Program) (hs : Schedule M 1 ps) :
    RunSafe (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel
      bounds M 1 ps s := by
  apply runSafe_of_each
  intro p hp state
  have hr := (mem_grid.mp ((hs p).1 hp)).1
  exact RealForwardInner.original_all_safe input output bounds (setup M 1 p state) Y X M N B one h
    (by simpa [setup] using hr) hX hY

theorem forward_noninner_grid_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N K B C : Nat) (one : Bool)
    (h : RealForwardNonInner.ExecutionDomain one X Y N B)
    (hX : M*N*K≤bounds X) (hY : M*N*K≤bounds Y)
    (ps : List Program) (hs : Schedule M (RealAddress.rounds K C) ps) :
    RunSafe (softmax_kernel_non_inner input output Y X M N K B C one).eraseDType.toAlgKernel
      bounds M (RealAddress.rounds K C) ps s := by
  apply runSafe_of_each
  intro p hp state
  have hr := (mem_grid.mp ((hs p).1 hp)).1
  exact RealForwardNonInner.original_all_safe input output bounds
    (setup M (RealAddress.rounds K C) p state) Y X M N K B C one h
    (by simpa [setup] using hr) hX hY

theorem backward_inner_grid_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N B C : Nat) (one : Bool)
    (h : RealBackwardInner.ExecutionDomain one P G Y N B)
    (hP : M*N≤bounds P) (hG : M*N≤bounds G) (hY : M*N≤bounds Y) (ps : List Program) :
    RunSafe (softmax_backward_kernel_inner storage P G Y M N C B one).eraseDType.toAlgKernel
      bounds (RealAddress.rounds M C) 1 ps s := by
  apply runSafe_of_each
  intro p _ state
  exact RealBackwardInner.original_all_safe storage bounds
    (setup (RealAddress.rounds M C) 1 p state) P G Y M N B C one h hP hG hY

theorem backward_noninner_grid_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (one : Bool)
    (h : RealBackwardNonInner.ExecutionDomain one P G Y N B)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y)
    (ps : List Program) (hs : Schedule M (RealAddress.rounds K C) ps) :
    RunSafe (softmax_backward_kernel_non_inner storage P G Y M N K B C one).eraseDType.toAlgKernel
      bounds M (RealAddress.rounds K C) ps s := by
  apply runSafe_of_each
  intro p hp state
  have hr := (mem_grid.mp ((hs p).1 hp)).1
  exact RealBackwardNonInner.original_all_safe storage bounds
    (setup M (RealAddress.rounds K C) p state) P G Y M N K B C one h
    (by simpa [setup] using hr) hP hG hY

/-- A lane coordinate identifies its owner uniquely, including tail blocks. -/
theorem owner_unique {C p q i j : Nat} (hi : i<C) (hj : j<C)
    (he : p*C+i=q*C+j) : p=q ∧ i=j := by
  have hd := congrArg (fun x ↦ x/C) he
  have hm := congrArg (fun x ↦ x%C) he
  dsimp only at hd hm
  rw [(decode_inner hi).1,(decode_inner hj).1] at hd
  rw [(decode_inner hi).2,(decode_inner hj).2] at hm
  exact ⟨hd,hm⟩

/-- Flattened non-inner addresses uniquely identify row, reduction lane and
column. Combining this with owner_unique gives disjoint CTA-owned columns. -/
theorem noninner_coordinate_unique {N K r s n m k l : Nat}
    (hn : n<N) (hm : m<N) (hk : k<K) (hl : l<K)
    (he : r*N*K+n*K+k=s*N*K+m*K+l) : r=s ∧ n=m ∧ k=l := by
  have hr := congrArg (fun x ↦ x/K/N) he
  have hnm := congrArg (fun x ↦ x/K%N) he
  have hkl := congrArg (fun x ↦ x%K) he
  dsimp only at hr hnm hkl
  rw [(decode_noninner hn hk).1,(decode_noninner hm hl).1] at hr
  rw [(decode_noninner hn hk).2.1,(decode_noninner hm hl).2.1] at hnm
  rw [(decode_noninner hn hk).2.2,(decode_noninner hm hl).2.2] at hkl
  exact ⟨hr,hnm,hkl⟩

#axiomsClean runSafe_of_each
#axiomsClean forward_inner_grid_safe
#axiomsClean forward_noninner_grid_safe
#axiomsClean backward_inner_grid_safe
#axiomsClean backward_noninner_grid_safe
#axiomsClean owner_unique
#axiomsClean noninner_coordinate_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
