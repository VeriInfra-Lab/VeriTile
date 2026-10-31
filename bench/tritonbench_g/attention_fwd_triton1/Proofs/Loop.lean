import bench.tritonbench_g.attention_fwd_triton1.Proofs.StepFacts

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000

namespace Graph
def carry (c : Shape) (k v : Nat → Term α) (head : Nat) (ifcond : Bool) :
    Nat → Matrix α c.D c.D
  | 0 => fun _ => num 0
  | n+1 => nextCarry c k v head n (carry c k v head ifcond n) ifcond
def output (c : Shape) (q k v : Nat → Term α) (scale : ℝ) (head chunk : Nat)
    (ifcond : Bool) : Matrix α 32 c.D :=
  TermBody.output c q k v scale head chunk (carry c k v head ifcond chunk) ifcond
end Graph

namespace SharedLoop
open SharedStep
def initial (c : Shape) (head : Nat) (s : State (Term α)) : State (Term α) :=
  (s.setReg "i_bh" .int [] (fun _ => (head : Int))).setReg "b_h" .real [c.D,c.D] (fun _ => num 0)
def state [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) : Nat → State (Term α)
  | 0 => initial c head s
  | n+1 => finished c scale head n storeFlag ifcond
      (Graph.carry c (read s "k") (read s "v") head ifcond n)
      ((state c scale head storeFlag ifcond s n).setReg "i" .nat [] (fun _ => n))

theorem input [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (r : RegionName) (off : Nat) (ho : r≠"o") (hh : r≠"h") :
    (state c scale head storeFlag ifcond s n).mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih => rw [state,finished_input _ _ _ _ _ _ _ _ r off ho hh]; exact ih

theorem reads [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (r : RegionName) (ho : r≠"o") (hh : r≠"h") :
    read (state c scale head storeFlag ifcond s n) r=read s r := by
  funext off
  unfold SharedStep.read
  rw [input c scale head storeFlag ifcond s n r off ho hh]

theorem head_reg [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) :
    (state c scale head storeFlag ifcond s n).regs .int [] "i_bh"=some (fun _ => (head : Int)) := by
  induction n with
  | zero => simp [state,initial]
  | succ n ih => simp only [state,finished_head]; simpa using ih

theorem carry_reg [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) :
    (state c scale head storeFlag ifcond s n).regs .real [c.D,c.D] "b_h"=
      some (Graph.carry c (read s "k") (read s "v") head ifcond n) := by
  cases n with
  | zero => simp [state,initial,Graph.carry]
  | succ n =>
    simp only [state,finished_carry,Graph.carry]
    change some (nextCarry c (read (state c scale head storeFlag ifcond s n) "k")
      (read (state c scale head storeFlag ifcond s n) "v") head n
      (Graph.carry c (read s "k") (read s "v") head ifcond n) ifcond)=_
    rw [reads c scale head storeFlag ifcond s n "k" (by decide) (by decide),
      reads c scale head storeFlag ifcond s n "v" (by decide) (by decide)]

theorem iteration [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) :
    run algebra (Program.body c scale storeFlag ifcond)
      ((state c scale head storeFlag ifcond s n).setReg "i" .nat [] (fun _ => n))=
      some (state c scale head storeFlag ifcond s (n+1)) := by
  apply body_run
  · simpa using head_reg c scale head storeFlag ifcond s n
  · simp
  · simpa using carry_reg c scale head storeFlag ifcond s n

theorem range_trajectory [Inhabited A] (M : Algebra A) (body : List ComputeStmt)
    (states : Nat → State A) (start stop : Nat) (le : start≤stop)
    (next : ∀ n, n<stop → run M body ((states n).setReg "i" .nat [] (fun _ => n))=
      some (states (n+1))) :
    range M "i" start stop 1 body (states start)=some (states stop) := by
  rw [range]
  simp only [Nat.one_ne_zero,↓reduceIte]
  by_cases lt : start<stop
  · simp only [lt,↓reduceIte,next start lt,Option.bind]
    exact range_trajectory M body states (start+1) stop (by omega) next
  · have equal : start=stop := by omega
    simp [lt,equal]
termination_by stop-start
decreasing_by omega

theorem kernel_run [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (pid : s.pids 0=head) :
    FP.Structural.exec algebra (Program.kernel c scale storeFlag ifcond) s=
      some (state c scale head storeFlag ifcond s c.chunks) := by
  rw [Program.quoted]
  simp only [FP.Structural.exec,Program.code,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    algebra,Option.bind,pid]
  change (range algebra "i" 0 c.chunks 1 (Program.body c scale storeFlag ifcond)
    (state c scale head storeFlag ifcond s 0)).bind (fun t => some t)=_
  rw [range_trajectory algebra _ (state c scale head storeFlag ifcond s) 0 c.chunks
    (Nat.zero_le _) (fun n _ => iteration c scale head storeFlag ifcond s n)]
  rfl

#axiomsClean iteration
#axiomsClean kernel_run
end SharedLoop
#specNonCircular Graph.carry avoiding [FP.Structural.exec,SharedLoop.state]
#specNonCircular Graph.output avoiding [FP.Structural.exec,SharedLoop.state]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
