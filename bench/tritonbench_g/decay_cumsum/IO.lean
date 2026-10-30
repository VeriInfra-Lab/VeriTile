import bench.tritonbench_g.decay_cumsum.Proofs.KernelReal
import bench.tritonbench_g.decay_cumsum.Proofs.KernelCompiler
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Complete streamed program IO for all three FP32 source templates. The
backward outputs dqo/dko intentionally share their input regions. Each channel
covers the entire main or short-tail chunk and every padded feature lane.
Structural `.real` is the retained FP32 payload tag, not a change of precision.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.DecayIO
open VeriTile.Triton FP.Structural Geometry
noncomputable section

def inputs (kind : Kind) (h : Call) (p : Phase) : List IOChannel :=
  match kind with
  | .forward => [⟨.real,[p.span,h.BK],"g"⟩]
  | .prepare => [⟨.real,[p.span,h.BK],"q"⟩,⟨.real,[p.span,h.BK],"k"⟩,⟨.real,[p.span,h.BK],"g"⟩]
  | .backward => [⟨.real,[p.span,h.BK],"dqi"⟩,⟨.real,[p.span,h.BK],"dqo"⟩,
      ⟨.real,[p.span,h.BK],"dki"⟩,⟨.real,[p.span,h.BK],"dko"⟩,
      ⟨.real,[p.span,h.BK],"q"⟩,⟨.real,[p.span,h.BK],"k"⟩,⟨.real,[p.span,h.BK],"g"⟩]
def outputs (kind : Kind) (h : Call) (p : Phase) : List IOChannel :=
  match kind with
  | .forward => [⟨.real,[p.span,h.BK],"go"⟩]
  | .prepare => [⟨.real,[p.span,h.BK],"qg"⟩,⟨.real,[p.span,h.BK],"kg"⟩]
  | .backward => [⟨.real,[p.span,h.BK],"dqo"⟩,⟨.real,[p.span,h.BK],"dko"⟩,
      ⟨.real,[p.span,h.BK],"dg"⟩]

theorem input_shape (kind : Kind) (h : Call) (p : Phase) (i : Fin (inputs kind h p).length) :
    (inputs kind h p)[i].shape=[p.span,h.BK] := by
  cases kind <;> fin_cases i <;> rfl
 theorem input_dtype (kind : Kind) (h : Call) (p : Phase) (i : Fin (inputs kind h p).length) :
    (inputs kind h p)[i].dtype=.real := by
  cases kind <;> fin_cases i <;> rfl
 theorem output_shape (kind : Kind) (h : Call) (p : Phase) (i : Fin (outputs kind h p).length) :
    (outputs kind h p)[i].shape=[p.span,h.BK] := by
  cases kind <;> fin_cases i <;> rfl
 theorem output_dtype (kind : Kind) (h : Call) (p : Phase) (i : Fin (outputs kind h p).length) :
    (outputs kind h p)[i].dtype=.real := by
  cases kind <;> fin_cases i <;> rfl

def inputIndex (kind : Kind) (h : Call) (p : Phase) (i : Fin (inputs kind h p).length) :
    TileIndex (inputs kind h p)[i].shape → TileIndex [p.span,h.BK] :=
  cast (congrArg TileIndex (input_shape kind h p i))
def outputIndex (kind : Kind) (h : Call) (p : Phase) (i : Fin (outputs kind h p).length) :
    TileIndex (outputs kind h p)[i].shape → TileIndex [p.span,h.BK] :=
  cast (congrArg TileIndex (output_shape kind h p i))

def channelResult {α : Type} (kind : Kind) (h : Call) (p : Phase)
    (f : RegionName → Fin p.span → Fin h.BK → α) : ChannelValues α (outputs kind h p) :=
  fun o j => cast (congrArg (Value α) (output_dtype kind h p o).symm)
    (f (outputs kind h p)[o].buf (outputIndex kind h p o j).1 (outputIndex kind h p o j).2.1)

def address (h : Call) (p : Phase) (prog : Program h p) (t j : Nat) : Nat :=
  prog.row*h.headStride+prog.chunk*p.span*h.width+prog.feature*h.BK+j+t*h.width

def memoryIO (α : Type) (kind : Kind) (h : Call) (p : Phase) (prog : Program h p) : MemoryIO α where
  inputs := inputs kind h p
  outputs := outputs kind h p
  buffers := Roles.buffers kind
  read := fun i _ j => address h p prog
    (inputIndex kind h p i j).1.val (inputIndex kind h p i j).2.1.val
  inputMask := fun _ _ _ => True
  write := fun o _ j => address h p prog
    (outputIndex kind h p o j).1.val (outputIndex kind h p o j).2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := by intro i; exact Fin.elim0 i
  scratchMask := by intro i; exact Fin.elim0 i

def launch {α : Type} (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) : Invocation α where
  kernel := Kernels.source kind h p scale
  backend := FP.ProfiledRewrite.engine M
  pids := fun ax => if ax=0 then prog.feature else if ax=1 then prog.chunk
    else if ax=2 then prog.row else 0

/-- Source and logical payloads remain identical. Native observation is a
separate shared `Observes` judgment, so arbitrary old Node inputs are retained. -/
def io {α : Type} (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) : ProgramIO α α where
  toMemoryIO := memoryIO α kind h p prog
  schedule := fun _ => [launch M kind h p prog scale]
  encode := id
  observe := fun _ _ => id

def logicalRead {α : Type} [Inhabited α] (kind : Kind) (h : Call) (p : Phase)
    (vals : ChannelValues α (inputs kind h p)) (r : RegionName) (t : Nat) (j : Fin h.BK) : α :=
  if ht : t<p.span then
    match kind with
    | .forward => if r="g" then vals (0 : Fin 1) (⟨t,ht⟩,j,()) else default
    | .prepare => if r="q" then vals (0 : Fin 3) (⟨t,ht⟩,j,())
        else if r="k" then vals (1 : Fin 3) (⟨t,ht⟩,j,())
        else if r="g" then vals (2 : Fin 3) (⟨t,ht⟩,j,()) else default
    | .backward => if r="dqi" then vals (0 : Fin 7) (⟨t,ht⟩,j,())
        else if r="dqo" then vals (1 : Fin 7) (⟨t,ht⟩,j,())
        else if r="dki" then vals (2 : Fin 7) (⟨t,ht⟩,j,())
        else if r="dko" then vals (3 : Fin 7) (⟨t,ht⟩,j,())
        else if r="q" then vals (4 : Fin 7) (⟨t,ht⟩,j,())
        else if r="k" then vals (5 : Fin 7) (⟨t,ht⟩,j,())
        else if r="g" then vals (6 : Fin 7) (⟨t,ht⟩,j,()) else default
  else default

/-- Independent inclusive prefixes, gated Q/K and reverse gradient prefixes;
the original decimal 1.44269504 is retained, with no reciprocal-ln2 identity. -/
def realTarget (kind : Kind) (h : Call) (p : Phase) (scale : ℝ)
    (vals : ChannelValues ℝ (inputs kind h p)) : ChannelValues ℝ (outputs kind h p) :=
  channelResult kind h p (fun r t j => KernelReal.evaluate kind p.span scale
    (fun s n => logicalRead kind h p vals s n j) r t.val)

/-- All source primitive choices, FMA and operand order remain in the existing
instruction model. No floating rewrite or Real/native equality is assumed. -/
def nativeTarget (instructions : Native.Instructions) (kind : Kind) (h : Call)
    (p : Phase) (scale : Native.Word) (vals : ChannelValues Native.Node (inputs kind h p)) :
    ChannelValues Native.Word (outputs kind h p) :=
  channelResult kind h p (fun r t j => Native.evaluateWords instructions kind p.span scale
    (fun s n => (logicalRead kind h p vals s n j).value) r t.val)

def nativeObservation (kind : Kind) (h : Call) (p : Phase) :
    (o : Fin (outputs kind h p).length) → TileIndex (outputs kind h p)[o].shape →
      Native.Node → Native.Word := fun _ _ value => value.value

end
end VeriTile.Bench.Repairs.DecayCumsum.DecayIO

namespace VeriTile.Bench.Repairs.DecayCumsum.LogicalData
open VeriTile.Triton FP.Structural Geometry
noncomputable section

abbrev Matrix (h : Call) (p : Phase) := Fin p.span → Fin h.BK → ℝ
structure PrepareInputs (h : Call) (p : Phase) where
  q : Matrix h p
  k : Matrix h p
  g : Matrix h p
structure BackwardInputs (h : Call) (p : Phase) where
  dqi : Matrix h p
  dqo : Matrix h p
  dki : Matrix h p
  dko : Matrix h p
  q : Matrix h p
  k : Matrix h p
  g : Matrix h p
structure PrepareOutputs (h : Call) (p : Phase) where
  qg : Matrix h p
  kg : Matrix h p
structure BackwardOutputs (h : Call) (p : Phase) where
  dq : Matrix h p
  dk : Matrix h p
  dg : Matrix h p

def Inputs (kind : Kind) (h : Call) (p : Phase) : Type :=
  match kind with
  | .forward => Matrix h p
  | .prepare => PrepareInputs h p
  | .backward => BackwardInputs h p

def Outputs (kind : Kind) (h : Call) (p : Phase) : Type :=
  match kind with
  | .forward => Matrix h p
  | .prepare => PrepareOutputs h p
  | .backward => BackwardOutputs h p

def sample (xs : Matrix h p) (t : Nat) (j : Fin h.BK) : ℝ :=
  if ht : t < p.span then xs ⟨t,ht⟩ j else 0

def input_forward (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.inputs .forward h p) ≃ Inputs .forward h p where
  toFun := fun xs t j => xs ⟨0,by simp [DecayIO.inputs]⟩ (t,j,())
  invFun := fun xs => Fin.cases (fun j => xs j.1 j.2.1) (fun o => Fin.elim0 o)
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; rfl

def input_prepare (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.inputs .prepare h p) ≃ Inputs .prepare h p where
  toFun := fun xs => ⟨fun t j => xs ⟨0,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨1,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨2,by simp [DecayIO.inputs]⟩ (t,j,())⟩
  invFun := fun xs => (Fin.cases (fun j => xs.q j.1 j.2.1) (Fin.cases (fun j => xs.k j.1 j.2.1) (Fin.cases (fun j => xs.g j.1 j.2.1) (fun o => Fin.elim0 o))))
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def input_backward (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.inputs .backward h p) ≃ Inputs .backward h p where
  toFun := fun xs => ⟨fun t j => xs ⟨0,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨1,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨2,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨3,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨4,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨5,by simp [DecayIO.inputs]⟩ (t,j,()), fun t j => xs ⟨6,by simp [DecayIO.inputs]⟩ (t,j,())⟩
  invFun := fun xs => (Fin.cases (fun j => xs.dqi j.1 j.2.1) (Fin.cases (fun j => xs.dqo j.1 j.2.1) (Fin.cases (fun j => xs.dki j.1 j.2.1) (Fin.cases (fun j => xs.dko j.1 j.2.1) (Fin.cases (fun j => xs.q j.1 j.2.1) (Fin.cases (fun j => xs.k j.1 j.2.1) (Fin.cases (fun j => xs.g j.1 j.2.1) (fun o => Fin.elim0 o))))))))
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def output_forward (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.outputs .forward h p) ≃ Outputs .forward h p where
  toFun := fun xs t j => xs ⟨0,by simp [DecayIO.outputs]⟩ (t,j,())
  invFun := fun xs => Fin.cases (fun j => xs j.1 j.2.1) (fun o => Fin.elim0 o)
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; rfl

def output_prepare (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.outputs .prepare h p) ≃ Outputs .prepare h p where
  toFun := fun xs => ⟨fun t j => xs ⟨0,by simp [DecayIO.outputs]⟩ (t,j,()), fun t j => xs ⟨1,by simp [DecayIO.outputs]⟩ (t,j,())⟩
  invFun := fun xs => (Fin.cases (fun j => xs.qg j.1 j.2.1) (Fin.cases (fun j => xs.kg j.1 j.2.1) (fun o => Fin.elim0 o)))
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def output_backward (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.outputs .backward h p) ≃ Outputs .backward h p where
  toFun := fun xs => ⟨fun t j => xs ⟨0,by simp [DecayIO.outputs]⟩ (t,j,()), fun t j => xs ⟨1,by simp [DecayIO.outputs]⟩ (t,j,()), fun t j => xs ⟨2,by simp [DecayIO.outputs]⟩ (t,j,())⟩
  invFun := fun xs => (Fin.cases (fun j => xs.dq j.1 j.2.1) (Fin.cases (fun j => xs.dk j.1 j.2.1) (Fin.cases (fun j => xs.dg j.1 j.2.1) (fun o => Fin.elim0 o))))
  left_inv := by
    intro xs
    funext o j
    fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def inputEquiv (kind : Kind) (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.inputs kind h p) ≃ Inputs kind h p :=
  match kind with
  | .forward => input_forward h p
  | .prepare => input_prepare h p
  | .backward => input_backward h p

def outputEquiv (kind : Kind) (h : Call) (p : Phase) :
    ChannelValues ℝ (DecayIO.outputs kind h p) ≃ Outputs kind h p :=
  match kind with
  | .forward => output_forward h p
  | .prepare => output_prepare h p
  | .backward => output_backward h p

/-- Same main/tail invocation, padded lane window, and in-place dqo/dko graph. -/
def io (rest : Algebra ℝ) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) :
    LogicalProgramIO ℝ ℝ (Inputs kind h p) (Outputs kind h p) where
  core := DecayIO.io (KernelReal.algebra rest) kind h p prog scale
  inputs := inputEquiv kind h p
  outputs := outputEquiv kind h p

end
end VeriTile.Bench.Repairs.DecayCumsum.LogicalData
