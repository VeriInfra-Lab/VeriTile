import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Grid

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
def OutputRegion (s : Shape) (b : Buffers) (p : Address) : Prop :=
  p.1=b.out ∧ b.outOffset≤p.2 ∧ p.2<b.outOffset+s.M*s.N
theorem output_inside (s : Shape) (b : Buffers) (x : MatrixCell s) : OutputRegion s b (outputAddress s b x) := by
  have h := Nat.mul_le_mul_right s.N x.1.isLt
  have hn := x.2.isLt
  simp only [Nat.succ_mul] at h
  exact ⟨rfl,by change b.outOffset≤b.outOffset+x.1.val*s.N+x.2.val; omega,
    by change b.outOffset+x.1.val*s.N+x.2.val<b.outOffset+s.M*s.N; omega⟩
theorem commit_full_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (v : H) (e : Commit s c) (m : Memory F H) (p : Address) (outside : ¬OutputRegion s b p) :
    commit a s c b v e m p=m p := by
  have different : p≠outputAddress s b e.1 := by
    intro h
    exact outside (h ▸ output_inside s b e.1)
  exact if_neg different
theorem gridRun_full_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (events : List (Commit s c)) (m : Memory F H)
    (p : Address) (outside : ¬OutputRegion s b p) : gridRun a s c b selected events m p=m p := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih => rw [gridRun,ih,commit_full_frame _ _ _ _ _ _ _ _ outside]
theorem reset_full_frame (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H)
    (p : Address) (outside : ¬OutputRegion s b p) : reset a s b m p=m p := if_neg outside
theorem launch_full_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (q : Schedule s c) (m : Memory F H)
    (p : Address) (outside : ¬OutputRegion s b p) : launch a s c b selected q m p=m p := by
  rw [launch,gridRun_full_frame _ _ _ _ _ _ _ _ outside,reset_full_frame _ _ _ _ _ outside]
theorem reset_idempotent (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) :
    reset a s b (reset a s b m)=reset a s b m := by
  funext p
  unfold reset
  split_ifs <;> rfl
theorem launch_reset (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (q : Schedule s c) (m : Memory F H) :
    launch a s c b selected q (reset a s b m)=launch a s c b selected q m := by
  simp only [launch,reset_idempotent]
#axiomsClean launch_full_frame
#axiomsClean reset_idempotent
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
