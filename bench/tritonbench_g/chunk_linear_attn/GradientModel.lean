import bench.tritonbench_g.chunk_linear_attn.OutputModel
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
def gradQuery (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) : Matrix α p.BK p.BT :=
  load M m "query" (pid.z*(p.T*p.K)) p.K p.T 1 p.K (pid.x*p.BK) (pid.y*p.BT)
def gradKey (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) : Matrix α p.BT p.BK :=
  load M m "key" (pid.z*(p.T*p.K)) p.T p.K p.K 1 (pid.y*p.BT) (pid.x*p.BK)
/-- Deliberately retain the source's combined NT*K bound. A padded key lane
can read the next history chunk; native zero-times-NaN behavior is preserved. -/
def gradHistory (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (vb : Nat) : Matrix α p.BV p.BK :=
  load M m "history" (pid.z*(p.NT*p.K*p.V)) p.V (p.NT*p.K) 1 p.V
    (vb*p.BV) (pid.y*p.K+pid.x*p.BK)
def gradHistoryGradient (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (vb : Nat) : Matrix α p.BK p.BV :=
  load M m "historyGradient" (pid.z*(p.NT*p.K*p.V)) (p.NT*p.K) p.V p.V 1
    (pid.y*p.K+pid.x*p.BK) (vb*p.BV)
def gradScore (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) : Matrix α p.BT p.BT := fun i =>
  if i.1.val ≤ i.2.1.val then mul M (dot D (gradKey M p pid m) (gradQuery M p pid m) i) (scalar M scale) else zero M
structure GradAcc (p : Parameters) (α : Type) where
  query : Matrix α p.BT p.BK
  key : Matrix α p.BT p.BK
  score : Matrix α p.BT p.BT
def gradInitial (M : Algebra α) (p : Parameters) : GradAcc p α :=
  ⟨fun _ => zero M,fun _ => zero M,fun _ => zero M⟩
def gradNext (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (vb : Nat) (a : GradAcc p α) : GradAcc p α :=
  ⟨fun i => add M (a.query i)
      (mul M (dot D (valueTile M p pid m "incoming" vb) (gradHistory M p pid m vb) i) (scalar M scale)),
   fun i => add M (a.key i)
      (dot D (valueTile M p pid m "value" vb) (transpose (gradHistoryGradient M p pid m vb)) i),
   fun i => add M (a.score i)
      (dot D (valueTile M p pid m "incoming" vb) (transpose (valueTile M p pid m "value" vb)) i)⟩
def gradStates (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) : Nat → GradAcc p α
  | 0 => gradInitial M p
  | v+1 => gradNext M D p pid m scale v (gradStates M D p pid m scale v)
def gradValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (vb : Nat) : Matrix α p.BT p.BV := fun i =>
  add M (dot D (gradKey M p pid m) (gradHistoryGradient M p pid m vb) i)
    (dot D (fun j => cast M (gradScore M D p pid m scale j)) (valueTile M p pid m "incoming" vb) i)
def gradScoreGradient (M : Algebra α) (p : Parameters) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BT := fun i =>
  cast M (if i.2.1.val ≤ i.1.val then mul M (a.score i) (scalar M scale) else zero M)
def gradQueryResult (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BK := fun i =>
  add M (a.query i) (dot D (gradScoreGradient M p scale a) (gradKey M p pid m) i)
def gradKeyResult (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (scale : ℝ) (a : GradAcc p α) : Matrix α p.BT p.BK := fun i =>
  add M (a.key i) (transpose (dot D (gradQuery M p pid m) (gradScoreGradient M p scale a)) i)
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
def gradientParts (p : Parameters) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (backwardQKV p scale).surfaceBody "i_v").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def GradientInputs (a b : Memory α) : Prop :=
  ∀ r : RegionName, r∈(["query","key","value","incoming","history","historyGradient"] : List RegionName) → ∀ o, a r o=b r o

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
  have hd : ("valuePartials" : RegionName) ∉ (["query","key","value","incoming","history","historyGradient"] : List RegionName) := by decide
  exact hd (he ▸ hr)
#specNonCircular gradientMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
