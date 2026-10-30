import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReal
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultNative
import bench.repairs.tritonbench_g.triton_conv2d_fwd.CandidateKernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural KernelNumeric
noncomputable section

/-- The original scalar/dot interpretation is retained on arbitrary symbolic
inputs; ordinary Real values and all FP32 words embed by `Value.scalar`.
Storage dtypes are FP32 in both inputs and output, independently of the chosen
FP16-operand / TF32 / scalar-FP32 computation path. -/
abbrev Payload := KernelNumeric.Value
abbrev InputLane (s : Shape) := TileIndex [s.n,s.channels,s.h,s.w]
abbrev WeightLane (s : Shape) := TileIndex [s.outputs,s.cg,s.kh,s.kw]
abbrev OutputLane (s : Shape) := TileIndex [s.groups,s.spatial,s.og]

def inputChannels (s : Shape) : List IOChannel :=
  [⟨.fp32,[s.n,s.channels,s.h,s.w],"input"⟩,
   ⟨.fp32,[s.outputs,s.cg,s.kh,s.kw],"weight"⟩]
def outputChannels (s : Shape) : List IOChannel :=
  [⟨.fp32,[s.groups,s.spatial,s.og],"output"⟩]
abbrev Inputs (A : Type) (s : Shape) := ChannelValues (Payload A) (inputChannels s)
abbrev Outputs (A : Type) (s : Shape) := ChannelValues (Payload A) (outputChannels s)

def inputAddress (s : Shape) (j : InputLane s) : Nat :=
  s.sn*j.1.val+s.sc*j.2.1.val+s.sx*j.2.2.1.val+s.sy*j.2.2.2.1.val

def weightAddress (s : Shape) (j : WeightLane s) : Nat :=
  s.so*j.1.val+s.tc*j.2.1.val+s.tx*j.2.2.1.val+s.ty*j.2.2.2.1.val

def outputCell (s : Shape) (j : OutputLane s) : Conv2d.Cell :=
  ⟨j.1.val,j.2.1.val,j.2.2.1.val⟩
def outputAddress (s : Shape) (j : OutputLane s) : Nat := outputPointer s (outputCell s j)

def readAddress (s : Shape) :
    (i : Fin (inputChannels s).length) → TileIndex (inputChannels s)[i].shape → Nat :=
  Fin.cases (inputAddress s) (Fin.cases (weightAddress s) (fun i => Fin.elim0 i))

def payload (s : Shape) (vals : Inputs A s) :
    (i : Fin (inputChannels s).length) → TileIndex (inputChannels s)[i].shape → Payload A :=
  Fin.cases (vals ⟨0,by simp [inputChannels]⟩)
    (Fin.cases (vals ⟨1,by simp [inputChannels]⟩) (fun i => Fin.elim0 i))

abbrev InputPosition (s : Shape) :=
  (i : Fin (inputChannels s).length) × TileIndex (inputChannels s)[i].shape

/-- A pure extension of finite logical input arrays to their strided addresses.
Overlapping coordinates are allowed: simultaneous input binding gives their
consistency. Neither execution nor an output value occurs in this extension. -/
def logicalRead (zero : Payload A) (s : Shape) (vals : Inputs A s)
    (r : RegionName) (off : Nat) : Payload A := by
  classical
  exact if h : ∃ pos : InputPosition s,
      (inputChannels s)[pos.1].buf=r ∧ readAddress s pos.1 pos.2=off then
    payload s vals (Classical.choose h).1 (Classical.choose h).2
  else zero

def logicalScalars (o : Operations A) (s : Shape) (vals : Inputs A s) : Memory A
  | .input,a => scalar o (logicalRead (.scalar o.zero) s vals "input" a)
  | .weight,a => scalar o (logicalRead (.scalar o.zero) s vals "weight" a)
  | _,_ => o.zero

/-- Full tensor windows with the actual positive input/weight strides; all
logical lanes are active and there is no global scratch or in-place output. -/
def memory (s : Shape) : MemoryIO (Payload A) where
  inputs := inputChannels s
  outputs := outputChannels s
  buffers := ["input","weight","output"]
  read := fun i _ => readAddress s i
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ => outputAddress s) (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

/-- Original quotation, original three PID axes and original local-register
ABI. The unchanged grid includes the redundant grouped-feature programs. -/
def invocation (o : Operations A) (s : Shape) (f : Flags) (pid : Program) : Invocation (Payload A) where
  kernel := KernelSource.kernel f.fp16 (KernelBinding.parameters s f)
  backend := KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f
  pids := fun k => if k=0 then pid.1 else if k=1 then pid.2.1 else if k=2 then pid.2.2 else 0

def program (o : Operations A) (s : Shape) (f : Flags) : ProgramIO (Payload A) (Payload A) where
  toMemoryIO := memory s
  schedule := fun _ => (programs s).map (invocation o s f)
  encode := id
  observe := fun _ _ => id

def realIO (s : Shape) (f : Flags) := program KernelReal.operations s f
def nativeIO (actual : KernelNative.Instructions) (s : Shape) (f : Flags) :=
  program (KernelNative.operations actual f) s f

/-- Independent grouped cross-correlation over the initial logical tensors. -/
def mathematical (s : Shape) (vals : Inputs ℝ s) : Outputs ℝ s :=
  Fin.cases (fun j => .scalar (Conv2d.mathematical s
    (logicalScalars KernelReal.operations s vals) (outputCell s j))) (fun i => Fin.elim0 i)

/-- Original ordered, padded K32 graph with the fixed K8/K16/scalar instruction
choices and casts. All word patterns and the original symbolic domain remain. -/
def nativeTarget (numerics : KernelNative.Instructions) (s : Shape) (f : Flags)
    (vals : Inputs Native.Word s) : Outputs Native.Word s :=
  Fin.cases (fun j => .scalar (reference (KernelNative.operations (KernelNative.software numerics) f)
    s f (logicalScalars (KernelNative.operations (KernelNative.software numerics) f) s vals)
    (outputCell s j))) (fun i => Fin.elim0 i)

/-- Default IEEE RN-even FP32 FMA circuit, with the original input payload
interpretation. No half conversion or tensor-instruction law is required. -/
def defaultTarget (actual : KernelNative.Instructions) (s : Shape) (f : Flags)
    (vals : Inputs Native.Word s) : Outputs Native.Word s :=
  Fin.cases (fun j => .scalar (KernelNative.defaultReference s
    (logicalScalars (KernelNative.operations actual f) s vals) (outputCell s j))) (fun i => Fin.elim0 i)

structure LogicalInputs (s : Shape) where
  input : Fin s.n → Fin s.channels → Fin s.h → Fin s.w → Payload ℝ
  weight : Fin s.outputs → Fin s.cg → Fin s.kh → Fin s.kw → Payload ℝ

abbrev LogicalOutputs (s : Shape) :=
  Fin s.groups → Fin s.n → Fin s.oh → Fin s.ow → Fin s.og → Payload ℝ

def logicalInputs (s : Shape) : Inputs ℝ s ≃ LogicalInputs s where
  toFun xs := ⟨fun n c h w => xs ⟨0,by simp [inputChannels]⟩ (n,c,h,w,()),
    fun o c h w => xs ⟨1,by simp [inputChannels]⟩ (o,c,h,w,())⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => xs.input i.1 i.2.1 i.2.2.1 i.2.2.2.1
    | ⟨1,_⟩ => fun i => xs.weight i.1 i.2.1 i.2.2.1 i.2.2.2.1
  left_inv := by intro xs; funext port i; fin_cases port <;> rcases i with ⟨a,b,c,d,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def spatialIndex (s : Shape) (n : Fin s.n) (h : Fin s.oh) (w : Fin s.ow) : Fin s.spatial :=
  finProdFinEquiv (n,finProdFinEquiv (h,w))

def logicalOutputs (s : Shape) : Outputs ℝ s ≃ LogicalOutputs s where
  toFun xs g n h w f := xs ⟨0,by simp [outputChannels]⟩ (g,spatialIndex s n h w,f,())
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i =>
      let nr : Fin s.n × Fin (s.oh*s.ow) := finProdFinEquiv.symm i.2.1
      let hw : Fin s.oh × Fin s.ow := finProdFinEquiv.symm nr.2
      xs i.1 nr.1 hw.1 hw.2 i.2.2.1
  left_inv := by
    intro xs; funext port i; fin_cases port
    rcases i with ⟨g,r,f,u⟩; cases u
    change xs ⟨0,by simp [outputChannels]⟩ (g,finProdFinEquiv ((finProdFinEquiv.symm r).1,
      finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm r).2)),f,PUnit.unit) = _
    rw [Equiv.apply_symm_apply,Equiv.apply_symm_apply]
  right_inv := by
    intro xs; funext g n h w f
    change xs g (finProdFinEquiv.symm (finProdFinEquiv (n,finProdFinEquiv (h,w)))).1
      (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv (n,finProdFinEquiv (h,w)))).2).1
      (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv (n,finProdFinEquiv (h,w)))).2).2 f = _
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]

def logicalRealIO (s : Shape) (flags : Flags) :
    LogicalProgramIO (Payload ℝ) (Payload ℝ) (LogicalInputs s) (LogicalOutputs s) where
  core := realIO s flags
  inputs := logicalInputs s
  outputs := logicalOutputs s

/-- Scalar projection of the original symbolic-payload and aliased-stride
interpretation. The logical arrays themselves retain every original payload. -/
def LogicalInputs.paddedInput (s : Shape) (xs : LogicalInputs s)
    (n c : Nat) (h w : Int) : ℝ :=
  inputArray s 0 (logicalScalars KernelReal.operations s ((logicalInputs s).symm xs)) n c h w

def LogicalInputs.filter (s : Shape) (xs : LogicalInputs s) (o c h w : Nat) : ℝ :=
  weightArray s (logicalScalars KernelReal.operations s ((logicalInputs s).symm xs)) o c h w

end
end VeriTile.Bench.Repairs.Conv2d.ConvIO

namespace VeriTile.Bench.Repairs.Conv2d.Pair
open VeriTile Triton FP.Structural KernelNumeric
noncomputable section

def originalKernel (s : Shape) (f : Flags) := KernelSource.kernel f.fp16 (KernelBinding.parameters s f)
def kernel (s : Shape) (f : Flags) := CandidateSource.kernel f.fp16 (KernelBinding.parameters s f)

def invocation (o : Operations A) (s : Shape) (f : Flags) (pid : Program) : Invocation (ConvIO.Payload A) :=
  { ConvIO.invocation o s f pid with kernel := kernel s f }

def variant (o : Operations A) (s : Shape) (f : Flags) : ProgramIO.Variant (ConvIO.program o s f) where
  schedule := fun _ => (programs s).map (invocation o s f)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.Conv2d.Pair
