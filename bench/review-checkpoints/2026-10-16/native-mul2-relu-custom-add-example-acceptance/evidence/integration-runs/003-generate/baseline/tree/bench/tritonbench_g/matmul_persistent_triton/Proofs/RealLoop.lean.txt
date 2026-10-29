import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealIteration
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealCarry
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopProgram

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoop
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000

def tileRow (p : Parameters) (pid q : Nat) :=
  Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM (Schedule.tileId p.SMS pid q)
def tileColumn (p : Parameters) (pid q : Nat) :=
  Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM (Schedule.tileId p.SMS pid q)

/-- Complement of exactly the active source output addresses assigned to this
program ID. It includes all other regions, masked tile lanes and stride gaps. -/
def Outside (p : Parameters) (pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  ∀ q, q<work p pid → ∀ j,
    RealStore.active p (tileRow p pid q) (tileColumn p pid q) j →
    ¬(p.C=r ∧ RealStore.offset p (tileRow p pid q) (tileColumn p pid q) j=o)

structure Holds (p : Parameters) (xs : RealMath.Inputs) (pid : Nat) (s₀ : BlockState)
    (i : Nat) (s : BlockState) : Prop where
  registers : RealLoopState.Holds p xs pid i s
  bound : i≤stop p pid
  input : RealInput.Represents p xs s
  frame : ∀ r o, Outside p pid r o → s.mem r o = s₀.mem r o

theorem initial (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (hinput : RealInput.Represents p xs s) :
    Holds p xs (s.pids 0) s 0 (RealPrefix.result p s) := by
  have hm : (RealPrefix.result p s).mem = s.mem := by
    have hf := congrArg BlockState.mem (RealPrefix.frame p s)
    exact hf
  refine ⟨RealLoopState.initial p xs s,Nat.zero_le _,?_,?_⟩
  · apply RealInput.preserved p xs s _ hinput
    all_goals intro o; rw [hm]
  · intro r o _
    rw [hm]

/-- The body step discharges the flat-loop invariant using the actual source
schedule and arithmetic chain; no body-execution premise remains. -/
theorem iteration (p : Parameters) (xs : RealMath.Inputs) (pid : Nat) (s₀ : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hk : 0<count p) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hA : p.C≠p.A) (hB : p.C≠p.B)
    (i : Nat) (s : BlockState) (hi : i<stop p pid) (h : Holds p xs pid s₀ i s) :
    ∃ t, stepStmts (RealProgram.body p) (s.setReg "_i" .nat [] (Tile.scalar i)) = some t ∧
      Holds p xs pid s₀ (i+1) t := by
  let entry := s.setReg "_i" .nat [] (Tile.scalar i)
  have hregisters := RealLoopState.set_index p xs pid i s h.registers
  have hinput : RealInput.Represents p xs entry :=
    RealInput.preserved p xs s entry h.input (fun _ => rfl) (fun _ => rfl)
  have he := RealIteration.executes p entry xs pid i hM hN hk hGM hs hp hi hinput hregisters
  have hr := RealIteration.ready p entry xs pid i hk hregisters
  refine ⟨RealIteration.result p entry xs pid i,he,?_,Nat.succ_le_of_lt hi,
    RealIteration.input_preserved p entry xs pid i hA hB hinput,?_⟩
  · exact RealCarry.preserves p (RealIteration.scheduled p entry pid i) xs pid i hk
      hr.fixed hr.coords hr.ki hr.tile
  · intro r o hout
    rw [RealIteration.frame p entry xs pid i r o]
    · exact h.frame r o hout
    · intro j hj
      exact hout (i/count p) (Schedule.iteration_bounds (count p) (work p pid) i hk hi).1 j hj

/-- The entire original positive-K source executes from an arbitrary initial
register file, preserves both original input arrays, and preserves every raw
memory cell outside the active output addresses assigned to this source PID.
All dimensions/strides/configuration remain symbolic and all K tails remain. -/
theorem executes (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hK : 0<p.K) (hBK : 0<p.BK)
    (hGM : 0<p.GM) (hs : 0<p.SMS) (hp : s.pids 0<p.SMS)
    (hA : p.C≠p.A) (hB : p.C≠p.B) (hinput : RealInput.Represents p xs s) :
    ∃ t, stepStmts (RealSource.kernel p Bool.false).body s = some t ∧
      Holds p xs (s.pids 0) s (stop p (s.pids 0)) t := by
  have hk : 0<count p := RealMath.blocks_positive p.K p.BK hK hBK
  exact RealLoopProgram.executes_from_invariant p s (Holds p xs (s.pids 0) s)
    (initial p xs s hinput) (fun _ _ h => h.bound)
    (iteration p xs (s.pids 0) s hM hN hk hGM hs hp hA hB)

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoop
