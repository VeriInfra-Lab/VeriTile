import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.token_attn_reduceV.Proofs.Real
import bench.tritonbench_g.token_attn_reduceV.Proofs.Stores
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.token_attn_reduceV.Kernels

/-! Logical input windows for the complete gathered PV grid. Runtime lengths
are ordinary metadata channels. `counts` describes their arbitrary logical
extents; its finite supremum only pads channel shapes, imposing no fixed cap.
Request, start and token values determine the actual source read addresses.
All source FP32 operations, discrete metadata and zero-block behavior remain. -/
namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

def capacity {B : Nat} (counts : Fin B → Nat) : Nat := Finset.univ.sup counts
theorem count_le_capacity {B : Nat} (counts : Fin B → Nat) (b : Fin B) : counts b≤capacity counts :=
  Finset.le_sup (Finset.mem_univ b)

def inputs (p : Parameters) (B H : Nat) (counts : Fin B → Nat) : List IOChannel :=
  [⟨.nat,[B],Region.cast p.length⟩,
   ⟨.nat,[B],Region.cast p.start⟩,
   ⟨.nat,[B],Region.cast p.request⟩,
   ⟨.nat,[B,capacity counts],Region.cast p.tokens⟩,
   ⟨.fp32,[B,H,capacity counts],p.prob⟩,
   ⟨.fp32,[B,H,capacity counts,p.width],p.value⟩]
def outputs (p : Parameters) (B H : Nat) : List IOChannel := [⟨.fp32,[B,H,p.width],p.output⟩]
abbrev Inputs (A : Type) (p : Parameters) (B H : Nat) (counts : Fin B → Nat) := ChannelValues A (inputs p B H counts)
abbrev Outputs (A : Type) (p : Parameters) (B H : Nat) := ChannelValues A (outputs p B H)

def length {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) : Nat := v ⟨0,by simp [inputs]⟩ (b,PUnit.unit)
def start {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) : Nat := v ⟨1,by simp [inputs]⟩ (b,PUnit.unit)
def request {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) : Nat := v ⟨2,by simp [inputs]⟩ (b,PUnit.unit)
def token {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) (n : Fin (capacity counts)) : Nat :=
  v ⟨3,by simp [inputs]⟩ (b,n,PUnit.unit)
def prob {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) (h : Fin H) (n : Fin (capacity counts)) : A :=
  v ⟨4,by simp [inputs]⟩ (b,h,n,PUnit.unit)
def val {A : Type} {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) (h : Fin H) (n : Fin (capacity counts)) (d : Fin p.width) : A :=
  v ⟨5,by simp [inputs]⟩ (b,h,n,d,PUnit.unit)

def read (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (q : Fin (inputs p B H counts).length) (v : Inputs A p B H counts) :
    TileIndex (inputs p B H counts)[q].shape → Nat :=
  match q with
  | ⟨0,_⟩ => fun j => j.1.val*p.lengthStride
  | ⟨1,_⟩ => fun j => j.1.val*p.startStride
  | ⟨2,_⟩ => fun j => j.1.val*p.requestStride
  | ⟨3,_⟩ => fun j => request v j.1*p.requestBatchStride+j.2.1.val*p.requestTokenStride
  | ⟨4,_⟩ => fun j => j.2.1.val*p.probHeadStride+(start v j.1+j.2.2.1.val)*p.probTokenStride
  | ⟨5,_⟩ => fun j => j.2.1.val/p.kvGroup*p.valueHeadStride+j.2.2.2.1.val*p.valueDimStride+
      token v j.1 j.2.2.1*p.valueBatchStride
  | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)

def inputMask (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (q : Fin (inputs p B H counts).length) : TileIndex (inputs p B H counts)[q].shape → Prop :=
  match q with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _ => 0<H
  | ⟨3,_⟩ => fun j => 0<H ∧ p.block≠0 ∧ j.2.1.val<counts j.1
  | ⟨4,_⟩ => fun j => p.block≠0 ∧ j.2.2.1.val<counts j.1
  | ⟨5,_⟩ => fun j => p.block≠0 ∧ j.2.2.1.val<counts j.1
  | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)

def outputCoord (p : Parameters) (B H : Nat) (q : Fin (outputs p B H).length) :
    TileIndex (outputs p B H)[q].shape → TileIndex [B,H,p.width] :=
  match q with
  | ⟨0,_⟩ => id
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (p : Parameters) (B H : Nat) (counts : Fin B → Nat) : MemoryIO A where
  inputs := inputs p B H counts
  outputs := outputs p B H
  scratch := []
  buffers := [Region.cast p.length,Region.cast p.start,Region.cast p.request,Region.cast p.tokens,p.prob,p.value,p.output]
  read := read p B H counts
  inputMask := fun q _ => inputMask p B H counts q
  write := fun q _ j =>
    let c := outputCoord p B H q j
    Grid.offset p (c.1.val,c.2.1.val) (c.2.2.1,PUnit.unit)
  outputMask := fun _ _ _ => True
  scratchWrite := fun q => Fin.elim0 q
  scratchMask := fun q => Fin.elim0 q

def invocation (M : Algebra A) (R : SumPrimitive A) (p : Parameters) (pid : Grid.PID) : Invocation A where
  kernel := kernel p
  backend := withSum M R
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2 else 0
  numPids := some (fun _ => 0)

def programIO (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID) : ProgramIO A A :=
  { memory A p B H counts with
    schedule := fun _ => jobs.map (invocation M R p)
    encode := id
    observe := fun _ _ => id }

/-- Complete candidate kernel with exactly the original invocation bindings. -/
def affineInvocation (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (pid : Grid.PID) : Invocation A :=
  { invocation M R p pid with kernel := affineKernel p }

def affine (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID) :
    ProgramIO.Variant (programIO M R p B H counts jobs) where
  schedule := fun _ => jobs.map (affineInvocation M R p)
  observe := fun _ _ => id

/-- Runtime metadata matches the logical extents. Counts are arbitrary;
capacity is derived from them, and no maximum length input guard is added. -/
def LengthsMatch (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs A p B H counts) : Prop := ∀ b, 0<H → length v b=counts b

def result {A : Type} {p : Parameters} {B H : Nat}
    (f : Fin B → Fin H → Fin p.width → A) : Outputs A p B H :=
  fun q => match q with
  | ⟨0,_⟩ => fun j => f j.1 j.2.1 j.2.2.1
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

def probAt {A : Type} [Inhabited A] {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) (h : Fin H) (n : Nat) : A :=
  if hn : n<capacity counts then prob v b h ⟨n,hn⟩ else default
def valAt {A : Type} [Inhabited A] {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (v : Inputs A p B H counts) (b : Fin B) (h : Fin H) (n : Nat) (d : Fin p.width) : A :=
  if hn : n<capacity counts then val v b h ⟨n,hn⟩ d else default

/-- Source-ordered recurrence on logical arrays: every padded product and
reduction node remains, as do the live accumulator and final FP32 cast. -/
def orderedStep (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (count : Nat) (xs : Nat → A) (ys : Nat → Fin p.width → A)
    (n : Nat) (previous : TileIndex [p.width] → A) : TileIndex [p.width] → A :=
  let product := fun j : TileIndex [p.block,p.width] =>
    M.binary (some .fp32) .fp32 .mul
      (if n+j.1.val<count then xs (n+j.1.val) else padding M)
      (if n+j.1.val<count then ys (n+j.1.val) j.2.1 else padding M)
  let blockValue := R.apply (some .fp32) ⟨0,by simp⟩ Bool.false product
  fun i => M.binary (some .fp32) .real .add (previous i)
    (M.cast (some .fp32) .fp32 .real (blockValue i))

def orderedRow (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (count : Nat) (xs : Nat → A) (ys : Nat → Fin p.width → A) (d : Fin p.width) : A :=
  M.cast none .real .fp32
    (foldRange (orderedStep M R p count xs ys) 0 count p.block
      (fun _ => M.literal (some .fp32) .real 0) (d,PUnit.unit))

def orderedTarget [Inhabited A] (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (v : Inputs A p B H counts) : Outputs A p B H :=
  result fun b h d => orderedRow M R p (counts b) (probAt v b h) (valAt v b h) d

end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Logical
open VeriTile Triton FP.Structural TokenAttnReduceV
noncomputable section

/-- Full logical tensors and all runtime metadata, including inactive entries. -/
structure Inputs (p : Parameters) (B H : Nat) (counts : Fin B → Nat) where
  length : Fin B → Nat
  start : Fin B → Nat
  request : Fin B → Nat
  token : Fin B → Fin (Interface.capacity counts) → Nat
  probability : Fin B → Fin H → Fin (Interface.capacity counts) → ℝ
  value : Fin B → Fin H → Fin (Interface.capacity counts) → Fin p.width → ℝ

def inputs (p : Parameters) (B H : Nat) (counts : Fin B → Nat) :
    Interface.Inputs ℝ p B H counts ≃ Inputs p B H counts where
  toFun := fun values => ⟨Interface.length values,Interface.start values,Interface.request values,
    Interface.token values,Interface.prob values,Interface.val values⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.length j.1
    | ⟨1,_⟩ => fun j => xs.start j.1
    | ⟨2,_⟩ => fun j => xs.request j.1
    | ⟨3,_⟩ => fun j => xs.token j.1 j.2.1
    | ⟨4,_⟩ => fun j => xs.probability j.1 j.2.1 j.2.2.1
    | ⟨5,_⟩ => fun j => xs.value j.1 j.2.1 j.2.2.1 j.2.2.2.1
    | ⟨n+6,h⟩ => False.elim (by simp [Interface.inputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j
    all_goals first | (rcases j with ⟨a,b,c,d,⟨⟩⟩; rfl) | (rcases j with ⟨a,b,c,⟨⟩⟩; rfl) | (rcases j with ⟨a,b,⟨⟩⟩; rfl) | (rcases j with ⟨a,⟨⟩⟩; rfl)
  right_inv := by intro xs; cases xs; rfl

def outputs (p : Parameters) (B H : Nat) :
    Interface.Outputs ℝ p B H ≃ (Fin B → Fin H → Fin p.width → ℝ) where
  toFun := fun values b h d => values ⟨0, by simp [Interface.outputs]⟩ (b,h,d,())
  invFun := Interface.result
  left_inv := by
    intro values
    funext port
    fin_cases port
    funext j
    rcases j with ⟨a,b,c,⟨⟩⟩
    rfl
  right_inv := by intro f; rfl

/-- Total finite-array lookup; input masks and the matched counts retain their original meaning. -/
def Inputs.probAt {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (xs : Inputs p B H counts) (b : Fin B) (h : Fin H) (n : Nat) : ℝ :=
  if hn : n < Interface.capacity counts then xs.probability b h ⟨n,hn⟩ else 0

def Inputs.valueAt {p : Parameters} {B H : Nat} {counts : Fin B → Nat}
    (xs : Inputs p B H counts) (b : Fin B) (h : Fin H) (n : Nat) (d : Fin p.width) : ℝ :=
  if hn : n < Interface.capacity counts then xs.value b h ⟨n,hn⟩ d else 0

def domain (p : Parameters) (B H : Nat) (counts : Fin B → Nat) (xs : Inputs p B H counts) : Prop :=
  ∀ b, 0<H → xs.length b = counts b

def io (p : Parameters) (B H : Nat) (counts : Fin B → Nat) (work : List Grid.PID) :
    LogicalProgramIO ℝ ℝ (Inputs p B H counts) (Fin B → Fin H → Fin p.width → ℝ) where
  core := Interface.programIO Real.algebra Real.sum p B H counts work
  inputs := inputs p B H counts
  outputs := outputs p B H

end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Logical
