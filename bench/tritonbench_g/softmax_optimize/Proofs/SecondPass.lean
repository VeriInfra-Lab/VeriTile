import bench.tritonbench_g.softmax_optimize.Proofs.OutputProperties
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_RealLoopFrame

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.SecondPass
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open Input Output OutputBodies OutputProperties
set_option maxHeartbeats 5000000
noncomputable section

inductive Mode where | full | tail

def active (mode : Mode) (N column : Nat) : Prop := match mode with
  | .full => True
  | .tail => column < N

def body (mode : Mode) (Y X : RegionName) (N B : Nat) : List Stmt := match mode with
  | .full => Source.outputFull Y X N B
  | .tail => Source.outputTail Y X N B

def result (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ) : BlockState := match mode with
  | .full => fullResult s Y X N B start row shift (Real.exp (-shift) * total)
  | .tail => tailResult s Y X N B start row shift (Real.exp (-shift) * total)

theorem run (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hm : s.regs .real [] "m" = some (Tile.scalar (shift : WithBot ℝ)))
    (hz : s.regs .real [] "z" = some (Tile.scalar ((Real.exp (-shift) * total : ℝ) : WithBot ℝ))) :
    stepStmts (body mode Y X N B) (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (result mode s Y X N B start row shift total) := by
  cases mode
  · exact OutputBodies.full_run _ _ _ _ _ _ _ _ _ hpid hcols hm hz
  · exact OutputBodies.tail_run _ _ _ _ _ _ _ _ _ hpid hcols hm hz

theorem keeps (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : kept name) :
    (result mode s Y X N B start row shift total).regs dtype shape name = s.regs dtype shape name := by
  cases mode
  · exact full_keeps _ _ _ _ _ _ _ _ _ _ _ _ hn
  · exact tail_keeps _ _ _ _ _ _ _ _ _ _ _ _ hn

theorem metadata_kept (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ) :
    (result mode s Y X N B start row shift total).pids = s.pids ∧
    (result mode s Y X N B start row shift total).numPids = s.numPids ∧
    (result mode s Y X N B start row shift total).undef = s.undef := by
  cases mode
  · exact full_metadata _ _ _ _ _ _ _ _ _
  · exact tail_metadata _ _ _ _ _ _ _ _ _

theorem frame (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ) (region : RegionName) (address : Nat)
    (outside : ∀ i : TileIndex [B], active mode N (start + i.1.val) →
      ¬ (Y = region ∧ row * N + (start + i.1.val) = address)) :
    (result mode s Y X N B start row shift total).mem region address = s.mem region address := by
  cases mode
  · exact full_frame _ _ _ _ _ _ _ _ _ _ _ (fun i => outside i True.intro)
  · exact tail_frame _ _ _ _ _ _ _ _ _ _ _ outside

theorem stores (mode : Mode) (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift total : ℝ) (i : TileIndex [B]) (hi : active mode N (start + i.1.val)) :
    (result mode s Y X N B start row shift total).readMem Y (row * N + (start + i.1.val)) =
      Real.exp (s.readMem X (row * N + (start + i.1.val))) / total := by
  cases mode
  · rw [result, full_readback]
    exact quotient_real _ shift total _ i rfl
  · rw [result, tail_readback _ _ _ _ _ _ _ _ _ i hi]
    apply quotient_real
    change start + i.1.val < N at hi
    simp only [tailLoad, hi, ↓reduceIte]

structure Stable (Y : RegionName) (N B : Nat) (shift total : ℝ)
    (initial current : BlockState) : Prop where
  pids : current.pids = initial.pids
  numPids : current.numPids = initial.numPids
  undef : current.undef = initial.undef
  pidReg : current.regs .int [] "pid_m" = some (Tile.scalar (initial.pid : Int))
  colsReg : current.regs .int [] "cols" = some (Tile.scalar (N : Int))
  previousReg : current.regs .int [] "prev_multiple" = some (Tile.scalar (Config.previous N B : Int))
  m : current.regs .real [] "m" = some (Tile.scalar (shift : WithBot ℝ))
  z : current.regs .real [] "z" = some (Tile.scalar ((Real.exp (-shift) * total : ℝ) : WithBot ℝ))
  frame : RowFrame Y N initial current

structure Invariant (Y X : RegionName) (N B : Nat) (shift total : ℝ)
    (initial : BlockState) (count : Nat) (current : BlockState) : Prop
    extends Stable Y N B shift total initial current where
  written : ∀ j : Fin N, j.val < count * B →
    current.readMem Y (initial.pid * N + j.val) =
      Real.exp (initial.readMem X (initial.pid * N + j.val)) / total

theorem Stable.input {Y X : RegionName} {N B : Nat} {shift total : ℝ} {initial current : BlockState}
    (h : Stable Y N B shift total initial current) (hYX : Y ≠ X) (offset : Nat) :
    current.readMem X offset = initial.readMem X offset := by
  unfold BlockState.readMem
  rw [h.frame X offset (Or.inl hYX.symm)]

theorem advance (mode : Mode) (Y X : RegionName) (N B count : Nat) (shift total : ℝ)
    (initial current : BlockState) (hYX : Y ≠ X)
    (h : Invariant Y X N B shift total initial count current)
    (safe : ∀ i : TileIndex [B], active mode N (count * B + i.1.val) → count * B + i.1.val < N) :
    Invariant Y X N B shift total initial (count + 1)
      (result mode current Y X N B (count * B) initial.pid shift total) := by
  have hmeta := metadata_kept mode current Y X N B (count * B) initial.pid shift total
  have hkeep := keeps mode current Y X N B (count * B) initial.pid shift total
  have hframe : RowFrame Y N initial
      (result mode current Y X N B (count * B) initial.pid shift total) := by
    intro region address outside
    rw [frame mode current Y X N B (count * B) initial.pid shift total region address]
    · exact h.frame region address outside
    · intro i hi he
      have hiN := safe i hi
      rcases outside with hr | hlo | hhi
      · exact hr he.1.symm
      · omega
      · omega
  refine ⟨⟨hmeta.1.trans h.pids, hmeta.2.1.trans h.numPids, hmeta.2.2.trans h.undef,
    ?_, ?_, ?_, ?_, ?_, hframe⟩, ?_⟩
  · rw [hkeep _ _ _ (by simp [kept])]; exact h.pidReg
  · rw [hkeep _ _ _ (by simp [kept])]; exact h.colsReg
  · rw [hkeep _ _ _ (by simp [kept])]; exact h.previousReg
  · rw [hkeep _ _ _ (by simp [kept])]; exact h.m
  · rw [hkeep _ _ _ (by simp [kept])]; exact h.z
  · intro j hj
    by_cases hold : j.val < count * B
    · have hf := frame mode current Y X N B (count * B) initial.pid shift total Y
        (initial.pid * N + j.val) (by intro i _ he; omega)
      unfold BlockState.readMem
      rw [hf]
      exact h.written j hold
    · have hbase : count * B ≤ j.val := by omega
      let i : TileIndex [B] := (⟨j.val - count * B, by
        have hbound := hj
        rw [Nat.add_mul, Nat.one_mul] at hbound
        omega⟩, PUnit.unit)
      have hi : count * B + i.1.val = j.val := by dsimp [i]; omega
      have ha : active mode N (count * B + i.1.val) := by
        cases mode
        · exact True.intro
        · simpa only [active, hi] using j.isLt
      have hs := stores mode current Y X N B (count * B) initial.pid shift total i ha
      rw [hi, h.toStable.input hYX] at hs
      exact hs

theorem full_step (Y X : RegionName) (N B count : Nat) (shift total : ℝ)
    (initial current : BlockState) (hYX : Y ≠ X) (hN : 0 < N)
    (h : Invariant Y X N B shift total initial count current) (hc : count < Config.fullTiles N B) :
    ∃ next, stepStmts (Source.outputFull Y X N B)
      (current.setReg "start_n" .int [] (Tile.scalar ((count * B : Nat) : Int))) = some next ∧
      Invariant Y X N B shift total initial (count + 1) next := by
  exact ⟨result .full current Y X N B (count * B) initial.pid shift total,
    run .full _ _ _ _ _ _ _ _ _ h.pidReg h.colsReg h.m h.z,
    advance .full _ _ _ _ _ _ _ _ _ hYX h (fun i _ => Config.full_tile_active hN hc i.1.isLt)⟩

theorem tail_step (Y X : RegionName) (N B count : Nat) (shift total : ℝ)
    (initial current : BlockState) (hYX : Y ≠ X)
    (h : Invariant Y X N B shift total initial count current) :
    ∃ next, stepStmts (Source.outputTail Y X N B)
      (current.setReg "start_n" .int [] (Tile.scalar ((count * B : Nat) : Int))) = some next ∧
      Invariant Y X N B shift total initial (count + 1) next := by
  exact ⟨result .tail current Y X N B (count * B) initial.pid shift total,
    run .tail _ _ _ _ _ _ _ _ _ h.pidReg h.colsReg h.m h.z,
    advance .tail _ _ _ _ _ _ _ _ _ hYX h (fun _ hi => hi)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.SecondPass
