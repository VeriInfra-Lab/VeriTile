import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReverseState

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
noncomputable section

theorem reverseKey_at (p : Layout) (pid : ProgramId) (r lane : Nat) (hr : r<p.T) :
    reverseKeyOffset p pid r lane=keyOffset p pid (p.T-1-r) lane := by
  exact reverse_offset _ _ _ _ hr

theorem reverseValue_at (p : Layout) (pid : ProgramId) (r lane : Nat) (hr : r<p.T) :
    reverseValueOffset p pid r lane=valueOffset p pid (p.T-1-r) lane := by
  exact reverse_offset _ _ _ _ hr

theorem reverseDo_at (p : Layout) (pid : ProgramId) (r lane : Nat) (hr : r<p.T) :
    reverseDoOffset p pid r lane=doOffset p pid (p.T-1-r) lane := by
  exact reverse_offset _ _ _ _ hr

theorem reverseGradient_at (p : Layout) (pid : ProgramId) (r lane : Nat) (hr : r<p.T) :
    reverseGradientOffset p pid r lane=gradientOffset p pid (p.T-1-r) lane := by
  exact reverse_offset _ _ _ _ hr

theorem reverseOutput_at (p : Layout) (pid : ProgramId) (r lane : Nat) (hr : r<p.T) :
    reverseOutputOffset p pid r lane=outputOffset p pid (p.T-1-r) lane := by
  exact reverse_offset _ _ _ _ hr

theorem reverseKey_step (p : Layout) (pid : ProgramId) (r lane : Nat) :
    reverseKeyOffset p pid r lane-p.DK=reverseKeyOffset p pid (r+1) lane := by
  simp [reverseKeyOffset,Nat.sub_sub,Nat.add_mul]

theorem reverseValue_step (p : Layout) (pid : ProgramId) (r lane : Nat) :
    reverseValueOffset p pid r lane-p.DV=reverseValueOffset p pid (r+1) lane := by
  simp [reverseValueOffset,Nat.sub_sub,Nat.add_mul]

theorem reverseDo_step (p : Layout) (pid : ProgramId) (r lane : Nat) :
    reverseDoOffset p pid r lane-p.doTime=reverseDoOffset p pid (r+1) lane := by
  simp [reverseDoOffset,Nat.sub_sub,Nat.add_mul]

theorem reverseGradient_step (p : Layout) (pid : ProgramId) (r lane : Nat) :
    reverseGradientOffset p pid r lane-p.DK=reverseGradientOffset p pid (r+1) lane := by
  simp [reverseGradientOffset,Nat.sub_sub,Nat.add_mul]

theorem reverseOutput_step (p : Layout) (pid : ProgramId) (r lane : Nat) :
    reverseOutputOffset p pid r lane-p.DV=reverseOutputOffset p pid (r+1) lane := by
  simp [reverseOutputOffset,Nat.sub_sub,Nat.add_mul]

@[simp] theorem reverse_add_tagged (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) (q dout : α) :
    Staged.add a (reverseNode a s x final pid r key value)
      (Staged.multiply a (Staged.tagged .scaledQuery q) (Staged.atom dout)) =
      Staged.tagged .reverseState (a.fma q dout (programReverseCarry a s x final pid r key value)) := by
  by_cases hz : r=0 <;>
    simp [reverseNode,hz,Staged.add,Staged.multiply,Staged.tagged,Staged.atom,Staged.product]

#axiomsClean reverseKey_at
#axiomsClean reverseDo_at
#axiomsClean reverseKey_step
#axiomsClean reverse_add_tagged
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
