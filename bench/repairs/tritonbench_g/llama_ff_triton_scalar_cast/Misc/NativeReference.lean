import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.IO
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.NativeReference
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Words

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section

def xValue {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (row t : Nat) (k : Fin 64) : α :=
  if t*64+k.val<p.inner then xs (0 : Fin 7) (row*p.inner+(t*64+k.val)) else halfZero a

def rmsValue {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (t : Nat) (k : Fin 64) : α :=
  if p.byte then a.cast none .f8e5 .fp16
    (a.decode .f8e5 (Float8E5Bits.ofInt
      (if t*64+k.val<p.inner then xs (6 : Fin 7) (t*64+k.val) else 0)))
  else if t*64+k.val<p.inner then xs (3 : Fin 7) (t*64+k.val) else halfZero a

def weightValue {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (third : Bool) (column t : Nat) (k : Fin 64) : α :=
  if p.byte then a.cast none .f8e5 .fp16
    (a.decode .f8e5 (Float8E5Bits.ofInt
      (if t*64+k.val<p.inner then
        if third then xs (5 : Fin 7) (t*64+k.val+column*p.inner)
        else xs (4 : Fin 7) (t*64+k.val+column*p.inner) else 0)))
  else if t*64+k.val<p.inner then
    if third then xs (2 : Fin 7) (t*64+k.val+column*p.inner)
    else xs (1 : Fin 7) (t*64+k.val+column*p.inner) else halfZero a

def scaled {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (row t : Nat) (k : Fin 64) :=
  a.binary none .fp16 .mul (xValue a p xs row t k) (rmsValue a p xs t k)

/-- Fixed four-K16 instruction stream per K64 block, including all padded
lanes and the original live accumulator. Scalar ports are unchanged. -/
def accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (third : Bool) (row column : Nat) :=
  (NativeReference.stream (steps p)).foldl (fun old e => a.mma16
    (fragment (scaled a p xs row e.1) e.2)
    (fragment (weightValue a p xs third column e.1) e.2) old)
      (a.literal (some .fp32) .real 0)

def moment {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α)
    (row : Nat) (lane : Fin 64) :=
  (List.range (steps p)).foldl (fun old t => a.binary (some .fp32) .real .add old
    (a.powi (some .fp32) (a.cast (some .fp32) .fp16 .real (xValue a p xs row t lane)) 2))
      (a.literal (some .fp32) .real 0)

def normalization {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters) (xs : Inputs α) (row : Nat) :=
  a.unary (some .fp32) .rsqrt (a.binary (some .fp32) .real .add
    (a.binary (some .fp32) .real .div
      (LlamaFF.butterfly64.tree.eval (a.binary (some .fp32) .real .add) (moment a p xs row))
      (a.fromInt (some .fp32) p.inner)) (a.literal (some .fp32) .real p.epsilon))

def nativeValue {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (row column : Nat) :=
  let rho := normalization a p xs row
  let w1 := accumulation a p xs Bool.false row column
  let w3 := accumulation a p xs Bool.true row column
  let mul := a.binary (some .fp32) .real .mul
  a.cast none .real .fp16 (mul (mul w3 rho) (mul (mul w1 rho)
    (g.sigmoidNeg (g.fma (g.neg w1) rho (a.literal (some .fp32) .real 0)))))

def nativeFormula (a : Words.Instructions) (p : Parameters) (xs : Inputs Words.F) :
    ChannelValues Words.H (outputPorts p) :=
  fun ⟨0,_⟩ j => Words.half (nativeValue (Words.arithmetic a) (Words.gates a) p xs j.1.val j.2.1.val)

#specNonCircular nativeFormula avoiding [source,run,Grid.grid,Grid.one,nativeExecution,gateExecution]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference
