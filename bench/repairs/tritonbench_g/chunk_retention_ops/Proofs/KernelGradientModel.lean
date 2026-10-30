import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardModel
import bench.repairs.tritonbench_g.chunk_linear_attn.GradientModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul load rawLoad dot scatter scatter_frame transpose valueBlocks)
noncomputable section
abbrev Memory := KernelBackward.Memory

def kernel (p : Parameters) (scale : ℝ) : ComputeKernel :=
  Source.backwardQKV "q" "k" "v" "h" "incoming" "dhistory" "queryGradient" "keyGradient" "valuePartials"
    (p.T*p.K) p.K 1 (p.T*p.V) p.V 1 (p.NT*p.K*p.V) p.V
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT

variable {α : Type} [Inhabited α]
def gradQuery (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) : Matrix α p.BK p.BT :=
  load M m "q" (pid.z*(p.T*p.K)) p.K p.T 1 p.K (pid.x*p.BK) (pid.y*p.BT)
def gradKey (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) : Matrix α p.BT p.BK :=
  load M m "k" (pid.z*(p.T*p.K)) p.T p.K p.K 1 (pid.y*p.BT) (pid.x*p.BK)
def gradHistory (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (vb : Nat) : Matrix α p.BV p.BK :=
  load M m "h" (pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V) p.V p.K 1 p.V
    (vb*p.BV) (pid.x*p.BK)
def gradHistoryGradient (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (vb : Nat) : Matrix α p.BK p.BV :=
  load M m "dhistory" (pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V) p.K p.V p.V 1
    (pid.x*p.BK) (vb*p.BV)
def valueTile (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (r : RegionName) (vb : Nat) : Matrix α p.BT p.BV :=
  load M m r (pid.z*(p.T*p.V)) p.T p.V p.V 1 (pid.y*p.BT) (vb*p.BV)
def keyDecay (M : Algebra α) (p : Parameters) (pid : PID) (i : Nat) : α :=
  M.unary (some .fp32) .exp2 (mul M
    (M.fromInt (some .fp32) ((min p.BT (p.T-pid.y*p.BT):Nat)-(i:Int)-1))
    (KernelBackward.logRate M p pid))
def queryDecay (M : Algebra α) (p : Parameters) (pid : PID) (scale : ℝ) (i : Nat) : α :=
  cast M (mul M (KernelBackward.weight M p pid i) (scalar M scale))
def scoreDecay (M : Algebra α) (p : Parameters) (pid : PID) (scale : ℝ) : Matrix α p.BT p.BT := fun i=>
  mul M (if i.2.1.val ≤ i.1.val then M.unary (some .fp32) .exp2
    (mul M (M.fromInt (some .fp32) ((i.1.val:Int)-i.2.1.val)) (KernelBackward.logRate M p pid))
    else zero M) (scalar M scale)
def gradScore (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) : Matrix α p.BT p.BT := fun i =>
  mul M (dot D (gradKey M p pid m) (gradQuery M p pid m) i) (transpose (scoreDecay M p pid scale) i)
structure GradAcc (p : Parameters) (α : Type) where
  query : Matrix α p.BT p.BK
  key : Matrix α p.BT p.BK
  score : Matrix α p.BT p.BT
def gradInitial (M : Algebra α) (p : Parameters) : GradAcc p α :=
  ⟨fun _ => zero M,fun _ => zero M,fun _ => zero M⟩
def gradNext (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (vb : Nat) (a : GradAcc p α) : GradAcc p α :=
  ⟨fun i => add M (a.query i)
      (dot D (valueTile M p pid m "incoming" vb) (gradHistory M p pid m vb) i),
   fun i => add M (a.key i)
      (dot D (valueTile M p pid m "v" vb) (transpose (gradHistoryGradient M p pid m vb)) i),
   fun i => add M (a.score i)
      (dot D (valueTile M p pid m "incoming" vb) (transpose (valueTile M p pid m "v" vb)) i)⟩
def gradStates (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) : Nat → GradAcc p α
  | 0 => gradInitial M p
  | v+1 => gradNext M D p pid m scale v (gradStates M D p pid m scale v)
def gradValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (vb : Nat) : Matrix α p.BT p.BV := fun i =>
  add M (mul M (dot D (gradKey M p pid m) (gradHistoryGradient M p pid m vb) i) (keyDecay M p pid i.1.val))
    (dot D (fun j => cast M (gradScore M D p pid m scale j)) (valueTile M p pid m "incoming" vb) i)
def gradScoreGradient (M : Algebra α) (p : Parameters) (pid : PID) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BT := fun i =>
  cast M (mul M (a.score i) (scoreDecay M p pid scale i))
def gradQueryResult (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BK := fun i =>
  add M (mul M (a.query i) (queryDecay M p pid scale i.1.val)) (dot D (gradScoreGradient M p pid scale a) (gradKey M p pid m) i)
def gradKeyResult (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BK := fun i =>
  add M (mul M (a.key i) (keyDecay M p pid i.1.val)) (transpose (dot D (gradQuery M p pid m) (gradScoreGradient M p pid scale a)) i)
def gradValueActive (p : Parameters) (pid : PID) (vb : Nat) (i : TileIndex [p.BT,p.BV]) :=
  pid.y*p.BT+i.1.val < p.T ∧ vb*p.BV+i.2.1.val < p.V
instance (p : Parameters) (pid : PID) (vb : Nat) : DecidablePred (gradValueActive p pid vb) := fun _ => inferInstanceAs (Decidable (_∧_))
def gradValueOffset (p : Parameters) (pid : PID) (heads vb : Nat) (i : TileIndex [p.BT,p.BV]) :=
  (pid.x*heads+pid.z)*(p.T*p.V)+(pid.y*p.BT+i.1.val)*p.V+(vb*p.BV+i.2.1.val)
def gradValueWrite (M : Algebra α) (p : Parameters) (pid : PID) (heads vb : Nat)
    (a : Matrix α p.BT p.BV) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BT,p.BV]) "valuePartials" (gradValueActive p pid vb)
    (gradValueOffset p pid heads vb) (fun i => .mk .real (cast M (a i))) m
def gradKeyActive (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BK]) :=
  pid.y*p.BT+i.1.val < p.T ∧ pid.x*p.BK+i.2.1.val < p.K
instance (p : Parameters) (pid : PID) : DecidablePred (gradKeyActive p pid) := fun _ => inferInstanceAs (Decidable (_∧_))
def gradKeyOffset (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BK]) :=
  pid.z*(p.T*p.K)+(pid.y*p.BT+i.1.val)*p.K+(pid.x*p.BK+i.2.1.val)
def gradKeyWrite (M : Algebra α) (p : Parameters) (pid : PID) (r : RegionName)
    (a : Matrix α p.BT p.BK) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BT,p.BK]) r (gradKeyActive p pid)
    (gradKeyOffset p pid) (fun i => .mk .real (cast M (a i))) m

def gradientMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (heads : Nat) (input current : Memory α) (scale : ℝ) : Memory α :=
  let a := gradStates M D p pid input scale (valueBlocks p)
  gradKeyWrite M p pid "keyGradient" (gradKeyResult M D p pid input scale a)
    (gradKeyWrite M p pid "queryGradient" (gradQueryResult M D p pid input scale a)
      (foldRange (fun vb mem => gradValueWrite M p pid heads vb (gradValue M D p pid input scale vb) mem)
        0 (valueBlocks p) 1 current))
def gradientParts (p : Parameters) (scale : ℝ) : Triton.Source.DynamicRange :=
  (Triton.Source.dynamicRange? (kernel p scale).surfaceBody "i_v").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def GradientInputs (a b : Memory α) : Prop :=
  ∀ r : RegionName, r∈(["q","k","v","incoming","h","dhistory"] : List RegionName) → ∀ o, a r o=b r o

theorem GradientInputs.refl (m : Memory α) : GradientInputs m m := by intro r hr o; rfl
theorem GradientInputs.trans {a b c : Memory α} (ab : GradientInputs a b) (bc : GradientInputs b c) : GradientInputs a c := by
  intro r hr o
  exact (ab r hr o).trans (bc r hr o)
theorem gradValueWrite_inputs (M : Algebra α) (p : Parameters) (pid : PID) (heads vb : Nat)
    (a : Matrix α p.BT p.BV) (m : Memory α) : GradientInputs (gradValueWrite M p pid heads vb a m) m := by
  intro r hr o
  apply scatter_frame
  left
  intro he
  have hd : ("valuePartials" : RegionName) ∉ (["q","k","v","incoming","h","dhistory"] : List RegionName) := by decide
  exact hd (he ▸ hr)
#specNonCircular gradientMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
