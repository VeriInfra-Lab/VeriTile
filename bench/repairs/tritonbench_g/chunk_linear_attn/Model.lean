import bench.repairs.tritonbench_g.chunk_linear_attn.Kernels
import bench.repairs.tritonbench_g.chunk_delta_fwd.Memory
import bench.repairs.tritonbench_g.chunk_delta_fwd.Range
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev Memory (α : Type) := RegionName → Nat → Cell α
abbrev Matrix (α : Type) (m n : Nat) := TileIndex [m,n] → α
structure PID where
  (x y z : Nat)
abbrev HValues (p : Parameters) (α : Type) := Matrix α p.BK p.BV
variable {α : Type} [Inhabited α]
def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0
def scalar (M : Algebra α) (x : ℝ) : α := M.literal (some .fp32) .real x
def cast (M : Algebra α) (x : α) : α := M.cast (some .fp32) .real .real x
def add (M : Algebra α) (x y : α) : α := M.binary (some .fp32) .real .add x y
def mul (M : Algebra α) (x y : α) : α := M.binary (some .fp32) .real .mul x y
def dot (D : DotPrimitive α) {m k n : Nat} (a : Matrix α m k) (b : Matrix α k n) : Matrix α m n :=
  D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[]) a b

def rawLoad (M : Algebra α) (m : Memory α) (r : RegionName)
    (base rows cols sr sc ro co : Nat) {br bc : Nat} : Matrix α br bc := fun i =>
  if ro+i.1.val < rows ∧ co+i.2.1.val < cols then
    (m r (base+(ro+i.1.val)*sr+(co+i.2.1.val)*sc)).read .real else zero M
def load (M : Algebra α) (m : Memory α) (r : RegionName)
    (base rows cols sr sc ro co : Nat) {br bc : Nat} : Matrix α br bc :=
  fun i => M.fp32Load (rawLoad M m r base rows cols sr sc ro co i)
abbrev scatter := @VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter
abbrev scatter_frame := @VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter_frame
abbrev foldRange_succ := @VeriTile.Bench.Repairs.ChunkDeltaFwd.foldRange_succ
abbrev foldRange_zero := @VeriTile.Bench.Repairs.ChunkDeltaFwd.foldRange_zero
abbrev range_invariant := @VeriTile.Bench.Repairs.ChunkDeltaFwd.range_invariant

def stateKernel (p : Parameters) (scale : ℝ) (rev seed final : Bool) :=
  if rev then backwardH p scale seed final else forwardH p seed final
def loopName (rev : Bool) : RegName := if rev then "j" else "i_t"
def carryName (rev : Bool) : RegName := if rev then "b_dh" else "b_h"
def seedName (rev : Bool) : RegionName := if rev then "terminalGradient" else "initial"
def historyName (rev : Bool) : RegionName := if rev then "historyGradient" else "history"
def finalName (rev : Bool) : RegionName := if rev then "initialGradient" else "final"
def chunk (p : Parameters) (rev : Bool) (n : Nat) := if rev then p.NT-1-n else n

def stateParts (p : Parameters) (scale : ℝ) (rev seed final : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (stateKernel p scale rev seed final).surfaceBody (loopName rev)).getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def stateStorePrefix (p : Parameters) (scale : ℝ) (rev seed final : Bool) :=
  (stateParts p scale rev seed final).body.take (if rev then 5 else 4)
def stateUpdateSuffix (p : Parameters) (scale : ℝ) (rev seed final : Bool) :=
  (stateParts p scale rev seed final).body.drop (if rev then 5 else 4)

def stateInitial (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (rev seed : Bool) : HValues p α := fun i =>
  if seed then cast M (rawLoad M m (seedName rev) (pid.z*p.K*p.V)
    p.K p.V p.V 1 (pid.x*p.BK) (pid.y*p.BV) i) else zero M

def stateLeft (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (scale : ℝ) (rev : Bool) (n : Nat) : Matrix α p.BK p.BT := fun i =>
  if rev then cast M (mul M
    (load M m "query" (pid.z*(p.T*p.K)) p.K p.T 1 p.K
      (pid.x*p.BK) (chunk p rev n*p.BT) i) (scalar M scale))
  else load M m "key" (pid.z*(p.T*p.K)) p.K p.T 1 p.K
    (pid.x*p.BK) (chunk p rev n*p.BT) i

def stateRight (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (rev : Bool) (n : Nat) : Matrix α p.BT p.BV := fun i =>
  if rev then cast M (load M m "incoming" (pid.z*(p.T*p.V)) p.T p.V p.V 1
    (chunk p rev n*p.BT) (pid.y*p.BV) i)
  else load M m "value" (pid.z*(p.T*p.V)) p.T p.V p.V 1
    (chunk p rev n*p.BT) (pid.y*p.BV) i

def stateNext (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (rev : Bool) (n : Nat) (h : HValues p α) : HValues p α :=
  fun i => add M (h i) (dot D (stateLeft M p pid m scale rev n) (stateRight M p pid m rev n) i)
def stateValues (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (rev seed : Bool) : Nat → HValues p α
  | 0 => stateInitial M p pid m rev seed
  | n+1 => stateNext M D p pid m scale rev n (stateValues M D p pid m scale rev seed n)

def hActive (p : Parameters) (pid : PID) (i : TileIndex [p.BK,p.BV]) :=
  pid.x*p.BK+i.1.val < p.K ∧ pid.y*p.BV+i.2.1.val < p.V
instance (p : Parameters) (pid : PID) : DecidablePred (hActive p pid) := fun _ => inferInstanceAs (Decidable (_∧_))
def hOffset (p : Parameters) (pid : PID) (c : Nat) (i : TileIndex [p.BK,p.BV]) :=
  pid.z*(p.NT*p.K*p.V)+c*p.K*p.V+(pid.x*p.BK+i.1.val)*p.V+(pid.y*p.BV+i.2.1.val)
def fOffset (p : Parameters) (pid : PID) (i : TileIndex [p.BK,p.BV]) :=
  pid.z*p.K*p.V+(pid.x*p.BK+i.1.val)*p.V+(pid.y*p.BV+i.2.1.val)
def writeHistory (M : Algebra α) (p : Parameters) (pid : PID) (rev : Bool)
    (n : Nat) (h : HValues p α) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BK,p.BV]) (historyName rev) (hActive p pid)
    (hOffset p pid (chunk p rev n)) (fun i => .mk .real (cast M (h i))) m
def writeStateFinal (M : Algebra α) (p : Parameters) (pid : PID) (rev final : Bool)
    (h : HValues p α) (m : Memory α) : Memory α :=
  if final then scatter (TileShape.allIndices [p.BK,p.BV]) (finalName rev) (hActive p pid)
    (fOffset p pid) (fun i => .mk .real (cast M (h i))) m else m

def stateMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input current : Memory α) (scale : ℝ) (rev seed final : Bool) : Memory α :=
  writeStateFinal M p pid rev final (stateValues M D p pid input scale rev seed p.NT)
    (foldRange (fun n mem => writeHistory M p pid rev n
      (stateValues M D p pid input scale rev seed n) mem) 0 p.NT 1 current)

def StateInputs (a b : Memory α) : Prop :=
  ∀ r : RegionName, r∈(["query","key","value","incoming","initial","terminalGradient"] : List RegionName) → ∀ o, a r o=b r o
theorem StateInputs.refl (m : Memory α) : StateInputs m m := by intro r hr o; rfl
theorem StateInputs.trans {a b c : Memory α} (ab : StateInputs a b) (bc : StateInputs b c) : StateInputs a c := by
  intro r hr o
  exact (ab r hr o).trans (bc r hr o)

theorem history_inputs (M : Algebra α) (p : Parameters) (pid : PID) (rev : Bool)
    (n : Nat) (h : HValues p α) (m : Memory α) : StateInputs (writeHistory M p pid rev n h m) m := by
  intro r hr o
  apply scatter_frame
  left
  intro he
  have hd : historyName rev ∉ (["query","key","value","incoming","initial","terminalGradient"] : List RegionName) := by cases rev <;> decide
  exact hd (he ▸ hr)
theorem final_inputs (M : Algebra α) (p : Parameters) (pid : PID) (rev final : Bool)
    (h : HValues p α) (m : Memory α) : StateInputs (writeStateFinal M p pid rev final h m) m := by
  cases final
  · exact StateInputs.refl _
  · intro r hr o
    apply scatter_frame
    left
    intro he
    have hd : finalName rev ∉ (["query","key","value","incoming","initial","terminalGradient"] : List RegionName) := by cases rev <;> decide
    exact hd (he ▸ hr)

#specNonCircular stateValues avoiding [run,FP.Structural.exec]
#specNonCircular stateMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
