import bench.tritonbench_g.attention_fwd_triton1.Proofs.LoopOutput

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedGrid
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 8000000

/-- Source empty-launch condition. Definitionally identical to Empty,
without importing the optional host/contract proof stack into kernel readers. -/
def Empty (c : Shape) : Prop := c.T=0 ∨ c.heads=0
instance (c : Shape) : Decidable (Empty c) := inferInstanceAs (Decidable (c.T=0 ∨ c.heads=0))

def atProgram (c : Shape) (head : Nat) (s : State (Term α)) : State (Term α) :=
  { s with
    pids := fun ax => if ax = 0 then head else 0
    numPids := fun ax => if ax = 0 then c.heads else 1 }
def state [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) : Nat → State (Term α)
  | 0 => s
  | n+1 => SharedLoop.state c scale n storeFlag ifcond
      (atProgram c n (state c scale storeFlag ifcond s n)) c.chunks

/-- Actual shared interpretation of every program, threading the preceding
program's complete memory. No successful execution is an input premise. -/
def execute [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) : Nat → Option (State (Term α))
  | 0 => some s
  | n+1 => do
      let before ← execute c scale storeFlag ifcond s n
      FP.Structural.exec algebra (Program.kernel c scale storeFlag ifcond) (atProgram c n before)

theorem executes [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) :
    execute c scale storeFlag ifcond s n=some (state c scale storeFlag ifcond s n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [execute,ih]
    exact SharedLoop.kernel_run c scale n storeFlag ifcond _ (by simp [atProgram])

theorem input [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (r : RegionName) (off : Nat) (ho : r≠"o") (hh : r≠"h") :
    (state c scale storeFlag ifcond s n).mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih => rw [state,SharedLoop.input _ _ _ _ _ _ _ r off ho hh]; exact ih

theorem reads [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (r : RegionName) (ho : r≠"o") (hh : r≠"h") :
    read (state c scale storeFlag ifcond s n) r=read s r := by
  funext off
  unfold SharedStep.read
  rw [input c scale storeFlag ifcond s n r off ho hh]

theorem output_earlier (c : Shape) (head later row j : Nat)
    (hh : head<later) (hr : row<c.T) (hj : j<c.D) :
    c.mainOffset head row j<later*c.headSize := by
  have rowBound := digit_bound row c.T j c.D hr hj
  have cellBound := digit_bound head later (row*c.D+j) c.headSize hh rowBound
  unfold Shape.mainOffset
  omega
theorem history_earlier (c : Shape) (head later chunk i j : Nat)
    (hh : head<later) (hc : chunk<c.chunks) (hi : i<c.D) (hj : j<c.D) :
    c.historyOffset head chunk i j<later*c.historyHead := by
  have rowBound := digit_bound chunk c.chunks i c.D hc hi
  have cellBound := digit_bound _ _ j c.D rowBound hj
  have headBound := digit_bound head later ((chunk*c.D+i)*c.D+j) c.historyHead hh cellBound
  unfold Shape.historyOffset
  omega

theorem output [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (head : Nat) (hh : head<n)
    (row : Nat) (hr : row<c.T) (j : Fin c.D) :
    (state c scale storeFlag ifcond s n).mem "o" (c.mainOffset head row j.val)=
      .mk .real (TermBody.cast (Graph.output c (read s "q") (read s "k") (read s "v") scale head
        (row/32) ifcond (⟨row%32,Nat.mod_lt _ (by decide)⟩,j,PUnit.unit))) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases earlier : head<n
    · rw [state,SharedLoop.output_before _ _ _ _ _ _ _ _ (output_earlier c head n row j.val earlier hr j.isLt)]
      exact ih earlier
    · have equal : head=n := by omega
      subst head
      have decomposition : row/32*32+row%32=row := by omega
      have out := SharedLoop.output c scale n storeFlag ifcond
        (atProgram c n (state c scale storeFlag ifcond s n)) c.chunks (row/32)
        (c.chunk_of_row row hr) ⟨row%32,Nat.mod_lt _ (by decide)⟩ j (by simpa [decomposition] using hr)
      change (state c scale storeFlag ifcond s (n+1)).mem "o"
        (mainAddress c n (row/32) (row%32) j.val)=_ at out
      simp only [mainAddress,decomposition] at out
      change _=Cell.mk .real (TermBody.cast (Graph.output c
        (read (state c scale storeFlag ifcond s n) "q")
        (read (state c scale storeFlag ifcond s n) "k")
        (read (state c scale storeFlag ifcond s n) "v") scale n (row/32) ifcond _)) at out
      rw [reads c scale storeFlag ifcond s n "q" (by decide) (by decide),
        reads c scale storeFlag ifcond s n "k" (by decide) (by decide),
        reads c scale storeFlag ifcond s n "v" (by decide) (by decide)] at out
      exact out

theorem history [Inhabited α] (c : Shape) (scale : ℝ) (ifcond : Bool)
    (s : State (Term α)) (n : Nat) (head : Nat) (hh : head<n)
    (chunk : Nat) (hc : chunk<c.chunks) (i j : Fin c.D) :
    (state c scale Bool.true ifcond s n).mem "h" (c.historyOffset head chunk i.val j.val)=
      .mk .real (TermBody.cast (Graph.carry c (read s "k") (read s "v") head ifcond chunk (i,j,PUnit.unit))) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases earlier : head<n
    · rw [state,SharedLoop.history_before _ _ _ _ _ _ _ _
        (history_earlier c head n chunk i.val j.val earlier hc i.isLt j.isLt)]
      exact ih earlier
    · have equal : head=n := by omega
      subst head
      have out := SharedLoop.history c scale n ifcond
        (atProgram c n (state c scale Bool.true ifcond s n)) c.chunks chunk hc i j
      change _=Cell.mk .real (TermBody.cast (Graph.carry c
        (read (state c scale Bool.true ifcond s n) "k")
        (read (state c scale Bool.true ifcond s n) "v") n ifcond chunk (i,j,PUnit.unit))) at out
      rw [reads c scale Bool.true ifcond s n "k" (by decide) (by decide),
        reads c scale Bool.true ifcond s n "v" (by decide) (by decide)] at out
      exact out

theorem frame [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (count : n≤c.heads) (r : RegionName) (off : Nat)
    (ho : r≠"o" ∨ c.mainSize≤off)
    (hh : storeFlag=Bool.false ∨ r≠"h" ∨ c.historySize≤off) :
    (state c scale storeFlag ifcond s n).mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [state,SharedLoop.frame _ _ _ _ _ _ (by omega) c.chunks (by rfl) r off ho hh]
    exact ih (by omega)

def result [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) : State (Term α) :=
  if Empty c then s else state c scale storeFlag ifcond s c.heads
def run [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) : Option (State (Term α)) :=
  if Empty c then some s else execute c scale storeFlag ifcond s c.heads

-- Materialize the shared equations in their owner; physical and logical IO
-- consumers must not generate duplicate declarations for these definitions.
abbrev result_unfold := @result.eq_1
abbrev run_unfold := @run.eq_1

theorem run_success [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term α)) :
    run c scale storeFlag ifcond s=some (result c scale storeFlag ifcond s) := by
  unfold run result
  split_ifs
  · rfl
  · exact executes c scale storeFlag ifcond s c.heads

#axiomsClean executes
#axiomsClean output
#axiomsClean history
#axiomsClean frame
#axiomsClean run_success
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedGrid
