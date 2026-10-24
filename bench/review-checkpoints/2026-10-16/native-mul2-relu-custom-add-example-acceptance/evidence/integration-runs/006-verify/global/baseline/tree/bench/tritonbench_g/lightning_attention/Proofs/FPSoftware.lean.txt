import bench.tritonbench_g.lightning_attention.Proofs.FPReference
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32

/-- Concrete binary32 scalar arithmetic and same-format physical/register
conversions. The real channel here is a promoted fp32 register, not a real
number and not native fp64. Only unused fields remain in rest. -/
def software (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with
    literal := fun p d r => if (p=none ∨ p=some .fp32) ∧ (d=.real ∨ d=.fp32) ∧ r=0
      then FP.Value.zero .fp32 else base.literal p d r
    cast := fun p src dst x =>
      if (p=some .fp32 ∧ src=.fp32 ∧ dst=.real) ∨
        ((p=none ∨ p=some .fp32) ∧ src=.real ∧ dst=.fp32) then x
      else base.cast p src dst x }

theorem software_cast_identity (rest : Algebra Word) (cfg : FP.Config) (x : Word) :
    (software rest cfg).cast (some .fp32) .fp32 .real x=x ∧
    (software rest cfg).cast none .real .fp32 x=x ∧
    (software rest cfg).cast (some .fp32) .real .fp32 x=x ∧
    (software rest cfg).cast none .fp32 .fp32 x=x := by
  simp [software,FP.Software.scalarFP32,FP.Software.fp32Context]

/-- Explicit software dot schedule: increasing k, separately rounded fp32
multiply then rounded fp32 add, starting at +0. It deliberately makes no
claim to equal a TF32/tensor-core/fused native dot. The accepted stages use
only batch=[] and default options; all intermediate results are bit values. -/
def sequentialDot (cfg : FP.Config) : (batch : TileShape) → {m k n : Nat} →
    Values Word .real (batch ++ [m,k]) → Values Word .real (batch ++ [k,n]) →
    Values Word .real (batch ++ [m,n])
  | [], _, k, _, a, b => fun i => (List.finRange k).foldl
      (fun acc j => FP.add cfg acc (FP.mul cfg (a (i.1,j,())) (b (j,i.2.1,()))))
      (FP.Value.zero .fp32)
  | _::tail, _, _, _, a, b => fun i => sequentialDot cfg tail (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def softwareDot (cfg : FP.Config) : Dot Word :=
  ⟨fun _ _ {batch} {_} {_} {_} a b => sequentialDot cfg batch a b⟩

def wordCarry [Inhabited Word] (p : Parameters) (s : State Word) : Values Word .real [p.keyDim,p.columns] :=
  fun i => (s.mem p.carry (kvOffset p i)).read .fp32

def wordKV [Inhabited Word] (cfg : FP.Config) (D : Dot Word) (p : Parameters) (s : State Word) :
    Values Word .fp32 [p.keyDim,p.columns] :=
  let k : Values Word .real [p.keyDim,p.rows] := fun i => (s.mem p.key (i.1.val*p.rows+i.2.1.val)).read .fp32
  let v : Values Word .real [p.rows,p.columns] := fun i => (s.mem p.value (i.1.val*p.columns+i.2.1.val)).read .fp32
  let update := D.apply (some .fp32) {} (batch := []) k v
  fun i => FP.add cfg (wordCarry p s i) (update i)

def wordInter [Inhabited Word] (D : Dot Word) (p : Parameters) (s : State Word) :
    Values Word .fp32 [p.rows,p.columns] :=
  let q : Values Word .real [p.rows,p.keyDim] := fun i => (s.mem p.query (i.1.val*p.keyDim+i.2.1.val)).read .fp32
  D.apply (some .fp32) {} (batch := []) q (wordCarry p s)

def wordQuery [Inhabited Word] (p : Parameters) (s : State Word) : Values Word .real [p.rows,p.keyDim] :=
  fun i => if p.blockIndex*p.rows+i.1.val<p.length then
    (s.mem p.query (s.pids 0*p.length*p.keyDim+(p.blockIndex*p.rows+i.1.val)*p.keyDim+i.2.1.val)).read .fp32
    else FP.Value.zero .fp32

def wordKey [Inhabited Word] (p : Parameters) (s : State Word) : Values Word .real [p.keyDim,p.rows] :=
  fun i => if p.blockIndex*p.rows+i.2.1.val<p.length then
    (s.mem p.key (s.pids 0*p.length*p.keyDim+(p.blockIndex*p.rows+i.2.1.val)*p.keyDim+i.1.val)).read .fp32
    else FP.Value.zero .fp32

def wordValue [Inhabited Word] (p : Parameters) (s : State Word) : Values Word .real [p.rows,p.columns] :=
  fun i => if bodyActive p i then (s.mem p.value (bodyOffset p s i)).read .fp32 else FP.Value.zero .fp32

def wordScore [Inhabited Word] (D : Dot Word) (p : Parameters) (s : State Word) : Values Word .real [p.rows,p.rows] :=
  let raw := D.apply (some .fp32) {} (batch := []) (wordQuery p s) (wordKey p s)
  fun i => if i.2.1.val ≤ i.1.val then raw i else FP.Value.zero .fp32

def wordBody [Inhabited Word] (cfg : FP.Config) (D : Dot Word) (p : Parameters) (s : State Word) :
    Values Word .fp32 [p.rows,p.columns] :=
  let intra := D.apply (some .fp32) {} (batch := []) (wordScore D p s) (wordValue p s)
  let inter := D.apply (some .fp32) {} (batch := []) (wordQuery p s) (wordCarry p s)
  fun i => FP.add cfg (intra i) (inter i)

def wordGrad [Inhabited Word] (p : Parameters) (s : State Word) : Values Word .fp32 [p.rows,p.columns] :=
  fun i => if gradActive p s i then (s.mem p.query (gradOffset p s i)).read .fp32 else FP.Value.zero .fp32

theorem kv_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) : kvValues (software rest cfg) D p s=wordKV cfg D p s := by
  funext i
  simp [kvValues,wordKV,carryValues,wordCarry,read32,software,FP.Software.scalarFP32,FP.Software.fp32Context]

theorem inter_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) : interValues (software rest cfg) D p s=wordInter D p s := by
  unfold interValues wordInter carryValues wordCarry read32
  funext i
  simp [interValues,wordInter,carryValues,wordCarry,read32,software,FP.Software.scalarFP32,FP.Software.fp32Context]

theorem body_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (s : State Word) : bodyValues (software rest cfg) D p s=wordBody cfg D p s := by
  unfold bodyValues wordBody bodyScore wordScore bodyQuery wordQuery bodyKey wordKey
    bodyValue wordValue carryValues wordCarry maskedRead32 read32
  simp [software,FP.Software.scalarFP32,FP.Software.fp32Context,show (0.0:ℝ)=0 by norm_num]

theorem grad_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) : gradValues (software rest cfg) p s=wordGrad p s := by
  funext i
  simp [gradValues,wordGrad,software,FP.Software.scalarFP32,FP.Software.fp32Context,
    show (0.0:ℝ)=0 by norm_num]
#axiomsClean software_cast_identity
#axiomsClean kv_software
#axiomsClean inter_software
#axiomsClean body_software
#axiomsClean grad_software
end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
