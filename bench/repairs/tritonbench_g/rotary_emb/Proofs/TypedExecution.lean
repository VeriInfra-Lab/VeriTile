import bench.repairs.tritonbench_g.rotary_emb.Kernels
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Model
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Pair

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def algebra {A : Type} (rest : Algebra A) (p : Ops A) : Algebra A :=
  { rest with
    fp32Load := id
    cast := fun _ _ _ => id
    binary := fun tag dtype op => match op with
      | .mul => p.mul
      | .sub => p.sub
      | .add => p.add
      | _ => rest.binary tag dtype op }

abbrev Lane (p : Kernels.Layout) := TileIndex [p.BLOCK_SEQ,p.BLOCK_HEAD,p.BLOCK_PAIRS]
def token {A : Type} (p : Kernels.Layout) (s : State A) (i : Lane p) :=
  s.pids 1*p.BLOCK_SEQ+i.1.val
def head {A : Type} (p : Kernels.Layout) (s : State A) (i : Lane p) :=
  s.pids 0*p.BLOCK_HEAD+i.2.1.val
def dimension (p : Kernels.Layout) (odd : Bool) (i : Lane p) :=
  i.2.2.1.val*2+if odd then 1 else 0
def active {A : Type} (p : Kernels.Layout) (s : State A) (key : Bool) (i : Lane p) : Prop :=
  token p s i<p.max_total_len ∧ head p s i<(if key then p.HEAD_K else p.HEAD_Q) ∧
    dimension p Bool.true i<p.ROTARY_DIM
instance {A : Type} (p : Kernels.Layout) (s : State A) (key : Bool) :
    DecidablePred (active p s key) := fun _ => inferInstanceAs (Decidable (_∧_∧_))
def region (p : Kernels.Layout) (key : Bool) : RegionName := if key then p.K else p.Q
def offset {A : Type} (p : Kernels.Layout) (s : State A) (key odd : Bool) (i : Lane p) : Nat :=
  token p s i*(if key then p.stride_kbs else p.stride_qbs)+
    head p s i*(if key then p.stride_kh else p.stride_qh)+
    dimension p odd i*(if key then p.stride_kd else p.stride_qd)
def coefficient {A : Type} (p : Kernels.Layout) (s : State A) (sine odd : Bool) (i : Lane p) : Nat :=
  token p s i*(if sine then p.stride_sinbs else p.stride_cosbs)+
    dimension p odd i*(if sine then p.stride_sind else p.stride_cosd)

/-- Independent pair operations over the entry memory. This is a direct
rotation formula, separate from the typed source and its evaluator. -/
def value {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.Layout) (s : State A)
    (key odd : Bool) (i : Lane p) : Cell A :=
  let a := (s.mem (region p key) (offset p s key Bool.false i)).read .real
  let b := (s.mem (region p key) (offset p s key Bool.true i)).read .real
  let c := (s.mem p.Cos (coefficient p s Bool.false odd i)).read .real
  let z := (s.mem p.Sin (coefficient p s Bool.true odd i)).read .real
  .mk .real (if odd then ops.add (ops.mul a z) (ops.mul b c)
    else ops.sub (ops.mul a c) (ops.mul b z))

def branch {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.Layout) (s : State A)
    (key : Bool) (entry : State A) : State A :=
  Pair.written (region p key) (active p s key) (offset p s key) (value ops p s key) entry

/-- Constructive success of the complete source, retaining all four stores
and the Q-before-K load/store order. -/
def execution {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.Layout) (s : State A) :
    {t : State A // FP.Structural.exec (algebra rest ops) (Kernels.kernel p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [Kernels.kernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    algebra,numericLt,natLt,bop,numeric,store,VeriTile.Triton.Region.cast,
    ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

#specNonCircular value avoiding [Kernels.kernel,sourcePair,execution]
#axiomsClean execution
end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
