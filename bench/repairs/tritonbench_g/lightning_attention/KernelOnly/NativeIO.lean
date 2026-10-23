import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativePhases
import VeriTile.Triton.Float.MaterializedProgramIO

/-! Native FP32 IO. Structural floating cells use the erased `.real` channel;
the source loads, arithmetic and stores remain FP32. Each CTA materializes its
actual source result before the next CTA starts. Forward's padded columns
retain the original E stride, even when they overlap later rows; no host
geometry contract is imposed here. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
noncomputable section

def launch (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) : MaterializedInvocation Word Native.Expr where
  source := invocation Native.algebra Native.dots stage d pid
  encode := Native.Expr.bits
  decodeAt := fun _ _ x => (Native.lower plan x).eval ops
  decode_encode := by intros; rfl

def schedule (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) : List (MaterializedInvocation Word Native.Expr) :=
  (grid stage d).map (launch ops plan stage d)

def paddedWidth (d : Host.Dimensions) : Nat := d.NV*d.BM

def forwardInputs (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.N,d.D],"query"⟩,
   ⟨.real,[d.B*d.H,d.N,d.D],"key"⟩,
   ⟨.real,[d.B*d.H,d.N,paddedWidth d],"value"⟩,
   ⟨.real,[d.B*d.H,d.N,paddedWidth d],"output"⟩]

def forwardOutputs (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.N,paddedWidth d],"output"⟩]

def backwardInputs (d : Host.Dimensions) : List IOChannel :=
  inputChannels d ++ backwardChannels d

/-- The final dimension is a window; row/head strides remain the source E. -/
def paddedIndex (d : Host.Dimensions)
    (i : TileIndex [d.B*d.H,d.N,paddedWidth d]) : Nat :=
  i.1.val*d.N*d.E+i.2.1.val*d.E+i.2.2.1.val

def forwardInputIndex (d : Host.Dimensions) (port : Fin (forwardInputs d).length) :
    TileIndex ((forwardInputs d)[port]).shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ => denseIndex
  | ⟨2,_⟩ | ⟨3,_⟩ => paddedIndex d

def backwardInputIndex (d : Host.Dimensions) (port : Fin (backwardInputs d).length) :
    TileIndex ((backwardInputs d)[port]).shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ | ⟨6,_⟩ => denseIndex

/-- Initial output words are explicit in-place channels, preserving the old
arbitrary-word domain. Exterior cells of every type remain unconstrained. -/
def forward (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    MaterializedProgramIO Word Word Native.Expr where
  inputs := forwardInputs d
  outputs := forwardOutputs d
  scratch := []
  buffers := ["query","key","value","output"]
  read := fun i _ => forwardInputIndex d i
  inputMask := fun _ _ _ => True
  write := fun ⟨0,_⟩ _ => paddedIndex d
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => schedule ops plan .forward d
  encode := id
  observe := fun _ _ => id

/-- Intra precedes inter. Inter reads the freshly materialized gradients. -/
def backward (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    MaterializedProgramIO Word Word Native.Expr where
  inputs := backwardInputs d
  outputs := backwardChannels d
  scratch := []
  buffers := ["query","key","value","incoming","queryGradient","keyGradient","valueGradient"]
  read := fun i _ => backwardInputIndex d i
  inputMask := fun _ _ _ => True
  write := fun i _ => backwardIndex d i
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => schedule ops plan .intra d ++ schedule ops plan .inter d
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeIO
