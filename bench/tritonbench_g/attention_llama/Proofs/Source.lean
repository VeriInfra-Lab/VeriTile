import bench.tritonbench_g.attention_llama.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Source scalar syntax retained until the selected native fusion and lane
context is known. Leaves retain native words; byte leaves retain all eight bits. -/
inductive Term (α : Type) where
  | leaf (value : α)
  | literal (value : ℝ)
  | negativeInfinity
  | add (a b : Term α)
  | sub (a b : Term α)
  | mul (a b : Term α)
  | div (a b : Term α)
  | maximum (a b : Term α)
  | exp (a : Term α)
  | half (a : Term α)
  | byte (value : UInt8)
  | dot (n : Nat) (a b : Fin n → Term α)
  | sum (n : Nat) (a : Fin n → Term α)
  | max (n : Nat) (a : Fin n → Term α)

instance : Inhabited (Term α) := ⟨.literal 0⟩

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values (Term α) .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : Term α :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i j)

def dotValues (left right out : FloatDType) : (batch : TileShape) → {m k n : Nat} →
    Values (Term α) left.toTileDType (batch++[m,k]) →
    Values (Term α) right.toTileDType (batch++[k,n]) →
    Values (Term α) out.toTileDType (batch++[m,n])
  | [],_,k,_,a,b,i => ofFloat out (.dot k
      (fun j=>toFloat left (a (i.1,j,()))) (fun j=>toFloat right (b (j,i.2.1,()))))
  | _::ds,_,_,_,a,b,i => dotValues left right out ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def algebra : Algebra (Term α) where
  literal := fun _ _ x => .literal x
  negInf := .negativeInfinity
  binary := fun _ _ op a b => match op with
    | .add => .add a b | .sub => .sub a b | .mul => .mul a b
    | .div => .div a b | .max => .maximum a b | _ => .literal 0
  unary := fun _ op a => match op with | .exp=>.exp a | _=>a
  cast := fun _ _ dst a => if dst=.fp16 then .half a else a
  fromNat := fun _ n => .literal n
  fromInt := fun _ n => .literal n
  fp32Bits := fun _ => .literal 0
  fp32Load := id
  float8E5Bits := some (fun _ b=>.byte (UInt8.ofNat b.val))
  reduceSum := fun _ {_} ax keep xs i => .sum _ (reductionInputs ax keep xs i)
  reduceMax := fun _ {_} ax keep xs i => .max _ (reductionInputs ax keep xs i)
  dot := fun _ {batch} {_} {_} {_} a b=>some (dotValues .real .real .real batch a b)
  dotTyped := fun _ l r o _ {batch} {_} {_} {_} a b=>some (dotValues l r o batch a b)

structure Config where
  (Q K V Out : RegionName)
  (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son : Nat)
  (NH H N SP D : Nat)
  (causal fp8 : Bool)

def kernel (c : Config) : ComputeKernel :=
  (if c.fp8 then TypedSource.bytes else TypedSource.half)
    c.Q c.K c.V c.Out c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk
    c.svz c.svh c.svk c.svn c.soz c.soh c.som c.son c.NH c.H c.N c.SP 64 64 c.D c.causal

def parts (c : Config) : Triton.Source.DynamicRange :=
  (Triton.Source.dynamicRange? (kernel c).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def bindScale (s : State (Term α)) (scale : Term α) : State (Term α) :=
  s.setReg "sm_scale" .real [] (fun _=>scale)

def entry (c : Config) (s : State (Term α)) (scale : Term α) :
    {t : State (Term α) // run algebra (parts c).before (bindScale s scale)=some t} := by
  cases hf : c.fp8 <;> cases hc : c.causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,hf,hc,TypedSource.half,TypedSource.bytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,bindScale]
    rfl

#axiomsClean entry
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
