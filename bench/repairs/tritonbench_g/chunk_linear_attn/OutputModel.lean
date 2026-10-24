import bench.repairs.tritonbench_g.chunk_linear_attn.Model
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]
def keyBlocks (p : Parameters) := (p.K+p.BK-1)/p.BK
def valueBlocks (p : Parameters) := (p.V+p.BV-1)/p.BV
def transpose {m n : Nat} (a : Matrix α m n) : Matrix α n m := fun i => a (i.2.1,i.1,())
def queryTile (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (kb : Nat) : Matrix α p.BT p.BK :=
  load M m "query" (pid.z*(p.T*p.K)) p.T p.K p.K 1 (pid.y*p.BT) (kb*p.BK)
def keyTile (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (kb : Nat) : Matrix α p.BK p.BT :=
  load M m "key" (pid.z*(p.T*p.K)) p.K p.T 1 p.K (kb*p.BK) (pid.y*p.BT)
def valueTile (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (r : RegionName) (vb : Nat) : Matrix α p.BT p.BV :=
  load M m r (pid.z*(p.T*p.V)) p.T p.V p.V 1 (pid.y*p.BT) (vb*p.BV)
def forwardHistoryTile (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (kb : Nat) : Matrix α p.BK p.BV :=
  load M m "history" (pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V)
    p.K p.V p.V 1 (kb*p.BK) (pid.x*p.BV)
structure OutputAcc (p : Parameters) (α : Type) where
  output : Matrix α p.BT p.BV
  score : Matrix α p.BT p.BT
def outputInitial (M : Algebra α) (p : Parameters) : OutputAcc p α :=
  ⟨fun _ => zero M,fun _ => zero M⟩
def outputNext (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (kb : Nat) (a : OutputAcc p α) : OutputAcc p α :=
  ⟨fun i => add M (a.output i) (dot D (queryTile M p pid m kb) (forwardHistoryTile M p pid m kb) i),
   fun i => add M (a.score i) (dot D (queryTile M p pid m kb) (keyTile M p pid m kb) i)⟩
def outputStates (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) : Nat → OutputAcc p α
  | 0 => outputInitial M p
  | k+1 => outputNext M D p pid m k (outputStates M D p pid m k)
def outputResult (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (a : OutputAcc p α) : Matrix α p.BT p.BV := fun i =>
  mul M (add M (a.output i)
    (dot D (fun j => cast M (if j.2.1.val ≤ j.1.val then a.score j else zero M))
      (valueTile M p pid m "value" pid.x) i)) (scalar M scale)
def outputActive (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BV]) :=
  pid.y*p.BT+i.1.val < p.T ∧ pid.x*p.BV+i.2.1.val < p.V
instance (p : Parameters) (pid : PID) : DecidablePred (outputActive p pid) := fun _ => inferInstanceAs (Decidable (_∧_))
def outputOffset (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BV]) :=
  pid.z*(p.T*p.V)+(pid.y*p.BT+i.1.val)*p.V+(pid.x*p.BV+i.2.1.val)
def outputWrite (M : Algebra α) (p : Parameters) (pid : PID)
    (a : Matrix α p.BT p.BV) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BT,p.BV]) "output" (outputActive p pid)
    (outputOffset p pid) (fun i => .mk .real (cast M (a i))) m
def outputMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input current : Memory α) (scale : ℝ) : Memory α :=
  outputWrite M p pid (outputResult M D p pid input scale (outputStates M D p pid input (keyBlocks p))) current
def outputParts (p : Parameters) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (forwardO p scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
#specNonCircular outputMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
