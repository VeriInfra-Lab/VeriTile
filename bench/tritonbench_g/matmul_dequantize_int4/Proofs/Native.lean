import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Setup
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
abbrev Half := MatmulDequantInt4.Binary16.Value
structure Instructions where
  mul : Half → Half → Half
  fma : Half → Half → Half → Half
  narrow : Word → Word
  mma : Opcode → ISA.MMA16
/-- Four local obligations only. The MMA port has fixed K=16, a finite opcode
family, and one result lane. There is no dot, reduction, execution, or output port. -/
structure Obligations (actual : Instructions) (target : Opcode → ISA.MMA16) : Prop where
  mul : ∀ x y,actual.mul x y=MatmulDequantInt4.Binary16.mul .ieeeValue x y
  fma : ∀ x y z,actual.fma x y z=FPCompiled.HalfFMA.fma .ieeeValue x y z
  narrow : ∀ x,actual.narrow x=QuantizeKvTransform.MixedBits.narrow .ieeeValue x
  mma : ∀ op a b acc,actual.mma op a b acc=target op a b acc

def scalar (actual : Instructions) : Algebra Word :=
  let base := FPSoftware.scalar FPCompiled.unused
  {base with
    binary := fun p d op x y =>
      if ((p=none ∨ p=some .fp32) ∧ d=.fp16) ∧ op=.mul then
        QuantizeKvTransform.MixedBits.embed (actual.mul (QuantizeKvTransform.MixedBits.half x)
          (QuantizeKvTransform.MixedBits.half y))
      else base.binary p d op x y
    cast := fun p src dst x =>
      match src,dst with
      | .real,.fp16 | .fp32,.fp16 => actual.narrow x
      | _,_ => base.cast p src dst x}

theorem scalar_eq (actual : Instructions) (target : Opcode → ISA.MMA16) (h : Obligations actual target) :
    scalar actual=FPSoftware.scalar FPCompiled.unused := by
  have hbin : (fun p d op x y =>
      if ((p=none ∨ p=some .fp32) ∧ d=.fp16) ∧ op=.mul then
        QuantizeKvTransform.MixedBits.embed (actual.mul (QuantizeKvTransform.MixedBits.half x)
          (QuantizeKvTransform.MixedBits.half y))
      else (FPSoftware.scalar FPCompiled.unused).binary p d op x y)=
      (FPSoftware.scalar FPCompiled.unused).binary := by
    funext p d op x y
    by_cases hc : ((p=none ∨ p=some .fp32) ∧ d=.fp16) ∧ op=.mul
    · rcases hc with ⟨⟨hp,rfl⟩,rfl⟩
      simp only [hp,and_true,ite_true,h.mul,FPSoftware.scalar,Int4Matmul.FPSoftware.scalar]
    · exact if_neg hc
  have hcast : (fun p src dst x =>
      match src,dst with
      | .real,.fp16 | .fp32,.fp16 => actual.narrow x
      | _,_ => (FPSoftware.scalar FPCompiled.unused).cast p src dst x)=
      (FPSoftware.scalar FPCompiled.unused).cast := by
    funext p src dst x
    cases src <;> cases dst <;> simp [h.narrow,FPSoftware.scalar,Int4Matmul.FPSoftware.scalar]
  simp only [scalar,hbin,hcast]

def fmaAssign (actual : Instructions) (p : Parameters) (current : State Word) : Option (State Word) := do
  let b ← current.regs .int [p.blockInner,p.blockColumns] "b"
  let scale ← current.regs .fp16 [p.blockColumns] "scales"
  let zero ← current.regs .fp16 [p.blockColumns] "zeros"
  pure (current.setReg "b" .fp16 [p.blockInner,p.blockColumns] (fun i =>
    QuantizeKvTransform.MixedBits.embed (actual.fma
      (MatmulDequantInt4.Binary16.round .ieeeValue (b i : Rat))
      (QuantizeKvTransform.MixedBits.half (scale (i.2.1,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.half (zero (i.2.1,PUnit.unit))).neg)))

def body (actual : Instructions) (op : Opcode) (p : Parameters) (k : Nat) (s : State Word) := do
  let start ← run (FPCompiled.backend (scalar actual) (ISA.dot (actual.mma op)))
    (FPFused.loopPrefix p) (s.setReg "k" .nat [] (fun _ => k))
  let mid ← fmaAssign actual p start
  run (FPCompiled.backend (scalar actual) (ISA.dot (actual.mma op))) (FPFused.suffix p) mid

def loop (actual : Instructions) (op : Opcode) (p : Parameters) (k : Nat) : Nat → State Word → Option (State Word)
  | 0,s => some s
  | n+1,s => (body actual op p k s).bind (loop actual op p (k+1) n)

def exec (actual : Instructions) (op : Opcode) (p : Parameters) (s : State Word) := do
  let start ← run (FPCompiled.backend (scalar actual) (ISA.dot (actual.mma op))) (FPCompiled.prelude p) s
  let count ← start.regs .nat [] "num_pid_k"
  let mid ← loop actual op p 0 (count PUnit.unit) start
  run (FPCompiled.backend (scalar actual) (ISA.dot (actual.mma op))) (FPCompiled.epilogue p) mid

theorem exec_transport (actual : Instructions) (target : Opcode → ISA.MMA16)
    (h : Obligations actual target) (op : Opcode) (p : Parameters) (s : State Word) :
    exec actual op p s=FPFused.exec (FPSoftware.scalar FPCompiled.unused) (ISA.dot (target op)) p s := by
  have hm : actual.mma op=target op := funext fun a => funext fun b => funext fun acc => h.mma op a b acc
  have hf : fmaAssign actual=FPFused.fmaAssign := by
    funext p s
    simp only [fmaAssign,FPFused.fmaAssign,h.fma]
  have hb : body actual op=FPFused.runBody (FPSoftware.scalar FPCompiled.unused) (ISA.dot (target op)) := by
    funext p k s
    simp only [body,FPFused.runBody,scalar_eq actual target h,hm,hf]
  have hl (k n : Nat) (s : State Word) : loop actual op p k n s=
      FPFused.loop (FPSoftware.scalar FPCompiled.unused) (ISA.dot (target op)) p k n s := by
    induction n generalizing k s with
    | zero => rfl
    | succ n ih =>
      simp only [loop,FPFused.loop,hb,ih]
      rfl
  simp only [exec,FPFused.exec,scalar_eq actual target h,hm,hl]

def grid (actual : Instructions) (c : Call) (i : Fin 17) (s : State Word) :=
  VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence
    (fun pid current => exec actual (tunings i).opcode (params c i)
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram current pid))
    (List.range (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.count (Grid.geometry (params c i)))) s

theorem grid_transport (actual : Instructions) (target : Opcode → ISA.MMA16)
    (h : Obligations actual target) (c : Call) (i : Fin 17) (s : State Word) :
    grid actual c i s=Grid.grid (target (tunings i).opcode) (params c i) s := by
  simp only [grid,Grid.grid,exec_transport actual target h]
  rfl

#axiomsClean scalar_eq
#axiomsClean exec_transport
#axiomsClean grid_transport
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Native
