import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedGeometry
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Native
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Candidate
import VeriTile.Triton.Float.ProgramEquivalence

/-! Public channels for one complete rotary source invocation. Each payload
channel is one interleaved half; coefficient channels broadcast across the
four head lanes, retaining their head-independent source mask. -/
namespace VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Lane (shape : Shape) := TileIndex [16,4,shape.pairBlock]

/-- FP32 storage and computation are fixed by `Kernels.kernel`; `.real` is
the Structural interpreter's erased floating cell type, for both semantics. -/
def inputChannels (shape : Shape) : List IOChannel :=
  [⟨.real,[16,4,shape.pairBlock],"Q"⟩,⟨.real,[16,4,shape.pairBlock],"Q"⟩,
   ⟨.real,[16,4,shape.pairBlock],"K"⟩,⟨.real,[16,4,shape.pairBlock],"K"⟩,
   ⟨.real,[16,4,shape.pairBlock],"Cos"⟩,⟨.real,[16,4,shape.pairBlock],"Cos"⟩,
   ⟨.real,[16,4,shape.pairBlock],"Sin"⟩,⟨.real,[16,4,shape.pairBlock],"Sin"⟩]

def outputChannels (shape : Shape) : List IOChannel :=
  [⟨.real,[16,4,shape.pairBlock],"Q"⟩,⟨.real,[16,4,shape.pairBlock],"Q"⟩,
   ⟨.real,[16,4,shape.pairBlock],"K"⟩,⟨.real,[16,4,shape.pairBlock],"K"⟩]

abbrev Inputs (A : Type) (shape : Shape) := ChannelValues A (inputChannels shape)
abbrev Outputs (A : Type) (shape : Shape) := ChannelValues A (outputChannels shape)

def payloadOffset (shape : Shape) (pids : Nat → Nat) (key odd : Bool) (j : Lane shape) :=
  (pids 1*16+j.1.val)*(if key then shape.headsK else shape.headsQ)*shape.width+
    (pids 0*4+j.2.1.val)*shape.width+j.2.2.1.val*2+if odd then 1 else 0

def coefficientOffset (shape : Shape) (pids : Nat → Nat) (sine odd : Bool)
    (j : Lane shape) :=
  (pids 1*16+j.1.val)*(if sine then shape.sinWidth else shape.cosWidth)+
    j.2.2.1.val*2+if odd then 1 else 0

def payloadActive (shape : Shape) (pids : Nat → Nat) (key : Bool) (j : Lane shape) : Prop :=
  pids 1*16+j.1.val<shape.tokens ∧
  pids 0*4+j.2.1.val<(if key then shape.headsK else shape.headsQ) ∧
  j.2.2.1.val*2+1<shape.rotary

def coefficientActive (shape : Shape) (pids : Nat → Nat) (j : Lane shape) : Prop :=
  pids 1*16+j.1.val<shape.tokens ∧ j.2.2.1.val*2+1<shape.rotary

def payload (values : Inputs A shape) (key odd : Bool) : Lane shape → A :=
  if key then
    if odd then values ⟨3,by simp [inputChannels]⟩ else values ⟨2,by simp [inputChannels]⟩
  else if odd then values ⟨1,by simp [inputChannels]⟩ else values ⟨0,by simp [inputChannels]⟩

def coefficient (values : Inputs A shape) (sine odd : Bool) : Lane shape → A :=
  if sine then
    if odd then values ⟨7,by simp [inputChannels]⟩ else values ⟨6,by simp [inputChannels]⟩
  else if odd then values ⟨5,by simp [inputChannels]⟩ else values ⟨4,by simp [inputChannels]⟩

/-- Independent interleaved rotation: each column has its own cosine and sine
coefficient; the two head counts and the two coefficient row strides may differ. -/
def pairFormula (ops : Ops A) (values : Inputs A shape) (key odd : Bool) (j : Lane shape) : A :=
  let x0 := payload values key Bool.false j
  let x1 := payload values key Bool.true j
  let c := coefficient values Bool.false odd j
  let z := coefficient values Bool.true odd j
  if odd then ops.add (ops.mul x0 z) (ops.mul x1 c)
  else ops.sub (ops.mul x0 c) (ops.mul x1 z)

def rotation (ops : Ops A) (shape : Shape) (values : Inputs A shape) : Outputs A shape :=
  Fin.cases (pairFormula ops values Bool.false Bool.false)
    (Fin.cases (pairFormula ops values Bool.false Bool.true)
      (Fin.cases (pairFormula ops values Bool.true Bool.false)
        (Fin.cases (pairFormula ops values Bool.true Bool.true) (fun o => Fin.elim0 o))))

/-- The independent selected mul/neg/fma graph, on every FP32 word and every
initial native node. No finite-value or numerical-conformance premise. -/
def nativePair (p : Native.Instructions) (values : Inputs Native.Node shape)
    (key odd : Bool) (j : Lane shape) : Native.Node :=
  Native.atom (Native.graph p
    (payload values key Bool.false j).value (payload values key Bool.true j).value
    (coefficient values Bool.false Bool.false j).value (coefficient values Bool.true Bool.false j).value
    (coefficient values Bool.false Bool.true j).value (coefficient values Bool.true Bool.true j).value odd)

def nativeRotation (p : Native.Instructions) (shape : Shape)
    (values : Inputs Native.Node shape) : Outputs Native.Node shape :=
  Fin.cases (nativePair p values Bool.false Bool.false)
    (Fin.cases (nativePair p values Bool.false Bool.true)
      (Fin.cases (nativePair p values Bool.true Bool.false)
        (Fin.cases (nativePair p values Bool.true Bool.true) (fun o => Fin.elim0 o))))

/-- All four output channels alias their corresponding input halves. The
coefficient windows broadcast over heads and remain live independently of Q/K
head masks. There is no private scratch or additional output footprint. -/
def memory (shape : Shape) (pids : Nat → Nat) : MemoryIO A where
  inputs := inputChannels shape
  outputs := outputChannels shape
  buffers := ["Q","K","Cos","Sin"]
  read := Fin.cases (fun _ => payloadOffset shape pids Bool.false Bool.false)
    (Fin.cases (fun _ => payloadOffset shape pids Bool.false Bool.true)
      (Fin.cases (fun _ => payloadOffset shape pids Bool.true Bool.false)
        (Fin.cases (fun _ => payloadOffset shape pids Bool.true Bool.true)
          (Fin.cases (fun _ => coefficientOffset shape pids Bool.false Bool.false)
            (Fin.cases (fun _ => coefficientOffset shape pids Bool.false Bool.true)
              (Fin.cases (fun _ => coefficientOffset shape pids Bool.true Bool.false)
                (Fin.cases (fun _ => coefficientOffset shape pids Bool.true Bool.true) (fun i => Fin.elim0 i))))))))
  inputMask := Fin.cases (fun _ => payloadActive shape pids Bool.false)
    (Fin.cases (fun _ => payloadActive shape pids Bool.false)
      (Fin.cases (fun _ => payloadActive shape pids Bool.true)
        (Fin.cases (fun _ => payloadActive shape pids Bool.true)
          (Fin.cases (fun _ => coefficientActive shape pids)
            (Fin.cases (fun _ => coefficientActive shape pids)
              (Fin.cases (fun _ => coefficientActive shape pids)
                (Fin.cases (fun _ => coefficientActive shape pids) (fun i => Fin.elim0 i))))))))
  write := Fin.cases (fun _ => payloadOffset shape pids Bool.false Bool.false)
    (Fin.cases (fun _ => payloadOffset shape pids Bool.false Bool.true)
      (Fin.cases (fun _ => payloadOffset shape pids Bool.true Bool.false)
        (Fin.cases (fun _ => payloadOffset shape pids Bool.true Bool.true) (fun o => Fin.elim0 o))))
  outputMask := Fin.cases (fun _ => payloadActive shape pids Bool.false)
    (Fin.cases (fun _ => payloadActive shape pids Bool.false)
      (Fin.cases (fun _ => payloadActive shape pids Bool.true)
        (Fin.cases (fun _ => payloadActive shape pids Bool.true) (fun o => Fin.elim0 o))))
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) : Invocation A where
  kernel := Kernels.kernel (Typed.layout shape)
  backend := Typed.algebra rest ops
  pids := pids

/-- The entire selected source invocation, including both Q and K and all four
stores. Launch geometry is explicit; the established complete Tensor grid is
retained in `TensorBridge` through the same execution foundation. -/
def kernelIO (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) : ProgramIO A A where
  toMemoryIO := memory shape pids
  schedule := fun _ => [invocation rest ops shape pids]
  encode := id
  observe := fun _ _ => id

/-- Existing integer-address candidate; its only change is commuted head-block
multiplication. The FP32 operations and the shared IO layout are identical. -/
def candidate (rest : Algebra A) (ops : Ops A) (shape : Shape) (pids : Nat → Nat) :
    ProgramIO.Variant (kernelIO rest ops shape pids) where
  schedule := fun _ => [{ invocation rest ops shape pids with
    kernel := Candidate.optimized (Typed.layout shape) }]
  observe := fun _ _ => id


/-- Original complete source invocation with its existing native observer. -/
def original (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (pids : Nat → Nat) :=
  ProgramIO.Implementation.original (kernelIO rest (Native.ops p) shape pids) (fun _ => True)

/-- Existing address-only candidate on the same eight-input/four-output IO. -/
def indexCommuted (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (pids : Nat → Nat) :=
  ProgramIO.Implementation.ofVariant (candidate rest (Native.ops p) shape pids) (fun _ => True)

/-- Named logical input families: Q/K halves and cosine/sine coefficients.
Every family retains its complete original tile, independently of masks. -/
structure RotationInputs (A : Type) (shape : Shape) where
  payload : Bool → Bool → Lane shape → A
  coefficient : Bool → Bool → Lane shape → A

def logicalInputs : Inputs A shape ≃ RotationInputs A shape where
  toFun := fun values => ⟨payload values,coefficient values⟩
  invFun := fun xs => Fin.cases (xs.payload Bool.false Bool.false)
    (Fin.cases (xs.payload Bool.false Bool.true) (Fin.cases (xs.payload Bool.true Bool.false)
      (Fin.cases (xs.payload Bool.true Bool.true) (Fin.cases (xs.coefficient Bool.false Bool.false)
        (Fin.cases (xs.coefficient Bool.false Bool.true) (Fin.cases (xs.coefficient Bool.true Bool.false)
          (Fin.cases (xs.coefficient Bool.true Bool.true) (fun i => i.elim0))))))))
  left_inv := by
    intro values
    funext channel
    fin_cases channel <;> rfl
  right_inv := by
    rintro ⟨values,coefficients⟩
    dsimp only
    congr 1 <;> funext first second <;> cases first <;> cases second <;> rfl

/-- Output coordinates directly name Q/K and first/second interleaved half. -/
def logicalOutputs : Outputs A shape ≃ (Bool → Bool → Lane shape → A) where
  toFun := fun values key odd =>
    if key then if odd then values (3 : Fin 4) else values (2 : Fin 4)
    else if odd then values (1 : Fin 4) else values (0 : Fin 4)
  invFun := fun f => Fin.cases (f Bool.false Bool.false)
    (Fin.cases (f Bool.false Bool.true) (Fin.cases (f Bool.true Bool.false)
      (Fin.cases (f Bool.true Bool.true) (fun i => i.elim0))))
  left_inv := by intro values; funext channel; fin_cases channel <;> rfl
  right_inv := by intro f; funext key odd; cases key <;> cases odd <;> rfl


/-- Named Q/K halves with the complete original invocation contract. -/
def logicalIO (rest : Algebra A) (ops : Ops A) (shape : Shape) (pids : Nat → Nat) :
    LogicalProgramIO A A (RotationInputs A shape) (Bool → Bool → Lane shape → A) where
  core := kernelIO rest ops shape pids
  inputs := logicalInputs
  outputs := logicalOutputs

end
end VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
