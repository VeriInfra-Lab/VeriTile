import bench.tritonbench_g.triton_mul2.Proofs.FinitePointerBoundary

namespace TritonMul2Finite
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
noncomputable section
open Classical

/-- n=2^31 is cast to the least signed i32 value. No signed lane can be
smaller, even if index multiplication wraps. The entire launch is masked. -/
theorem minimum_size_all_masked (block pid lane : Nat) :
    active32 (2^31) block pid lane=Bool.false := by
  have h := BitVec.le_toInt (offset32 block pid lane)
  have hn : (BitVec.ofNat 32 (2^31)).toInt= -(2^31) := by decide +kernel
  simp only [active32,hn,decide_eq_false_iff_not]
  norm_num at h ⊢
  omega

theorem load_false_mask (d : Storage) (p : Layout) (snapshot : ByteMemory) (pid : Nat)
    (inputOffset outputOffset : Nat → BitVec 32) (mask : Nat → Bool)
    (inactive : ∀ lane, mask lane=Bool.false) (lanes : List Nat) :
    loadTile d p snapshot pid inputOffset outputOffset mask lanes=.ok [] := by
  induction lanes with
  | nil => rfl
  | cons lane rest ih => simp [loadTile,inactive,ih]
theorem minimum_program_no_access (b : Numerics) (d : Storage) (p : Layout)
    (block pid : Nat) (state : State) :
    executeProgram b d p (2^31) block pid state=.ok state := by
  unfold executeProgram
  rw [load_false_mask d p state.memory pid _ _ _ (minimum_size_all_masked block pid)]
  rfl
theorem minimum_grid_no_access (b : Numerics) (d : Storage) (p : Layout)
    (block : Nat) (pids : List Nat) (state : State) :
    executeGrid b d p (2^31) block pids state=.ok state := by
  induction pids generalizing state with
  | nil => rfl
  | cons pid rest ih => simp only [executeGrid,minimum_program_no_access,bind_ok,ih]
theorem minimum_launch_no_access (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (block : Nat) (pids : List Nat) (initial : ByteMemory) :
    sourceLaunch b d p mode input output (2^31) block pids initial=.ok initial := by
  unfold sourceLaunch
  rw [source_grid_correspondence,minimum_grid_no_access]
  rfl
theorem zero_grid_no_access (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat) (initial : ByteMemory) :
    sourceLaunch b d p mode input output n block [] initial=.ok initial := rfl

/-- Successful source launches have one independently specified result for
every byte, including untouched output cells, padding, and input/exterior bytes. -/
def LaunchResult (b : Numerics) (d : Storage) (p : Layout) (n block : Nat)
    (pids : List Nat) (initial final : ByteMemory) : Prop :=
  ∃ writes, launchPlan b d p n block pids initial=.ok writes ∧
    (∀ address, final address=byteResult writes initial address) ∧
    (∀ w∈writes, p.access.backingByte p.access.currentDevice w.deviceAddress=some w.outputAddress) ∧
    (∀ address, ¬p.output.Contains address → final address=initial address)
theorem successful_launch_result (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat) (pids : List Nat)
    (initial final : ByteMemory)
    (h : sourceLaunch b d p mode input output n block pids initial=.ok final) :
    LaunchResult b d p n block pids initial final := by
  rw [launch_correspondence] at h
  cases hp : launchPlan b d p n block pids initial with
  | error issue => simp [hp,Except.map] at h
  | ok writes =>
    simp only [hp,Except.map,Except.ok.injEq] at h
    subst final
    exact ⟨writes,hp,fun _ => rfl,launch_write_resolution b d p n block pids initial writes hp,launch_frame b d p n block pids initial writes hp⟩

#specNonCircular LaunchResult avoiding [sourceLaunch,sourceGrid,executeGrid,executeProgram,applyWrites,source,Expr.eval]
#axiomsClean minimum_size_all_masked
#axiomsClean minimum_launch_no_access
#axiomsClean zero_grid_no_access
#axiomsClean successful_launch_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
