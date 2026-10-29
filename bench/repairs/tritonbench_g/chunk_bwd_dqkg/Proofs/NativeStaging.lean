import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.GeneralPlans
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedPhysical
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Source syntax is retained until its native fusion context is known.
Every materialized leaf is a scalar or fixed K8 instruction, never an entire
variable-K dot, reduction or output oracle. -/
inductive Stage where
  | ready (g : Graph)
  | zero
  | literal (x : ℝ)
  | add (a b : Stage)
  | sub (a b : Stage)
  | mul (a b : Stage)
  | minus (a : Stage)
  | exp (a : Stage)
  | dot (D row col : Nat) (a b : Fin D → Stage)
  | total (n : Nat) (xs : Fin n → Stage)
  | row (n : Nat) (xs : Fin n → Stage)

instance : Inhabited Stage := ⟨.zero⟩

def bounded {n : Nat} (xs : Fin n → Graph) (i : Nat) : Graph :=
  if h : i<n then xs ⟨i,h⟩ else zero

/-- Ordered lowering of the independently stated source recurrence. Reordering
and fusion are explicit structural rules, not deferred numerical equalities. -/
def correctionProduct (p : Descriptor) (g : Graph) : Graph :=
  match g with
  | .mul32 a b=>if p.swapCorrectionOperands then .mul32 b a else .mul32 a b
  | _=>g

def historyUpdate (p : Descriptor) (carry block : Graph) : Graph :=
  if p.historyCarryFirst then .add32 carry block else .add32 block carry

def lower (p : Descriptor) (runtime : Runtime) : Stage → Graph
  | .ready g=>g
  | .zero=>zero
  | .literal x=>.literal32 x
  | .add a (.dot D r c x y)=>dotTF32 D r c (fun j=>lower p runtime (x j)) (fun j=>lower p runtime (y j)) (lower p runtime a)
  | .add (.mul a b) (.total _ xs)=>.fma32 (lower p runtime a) (lower p runtime b)
      (p.correctionSum (bounded (fun j=>correctionProduct p (lower p runtime (xs j)))))
  | .add a (.total _ xs)=>historyUpdate p (lower p runtime a) (p.historySum (bounded (fun j=>lower p runtime (xs j))))
  | .add .zero (.row _ xs)=>.add32
      (rowSum p.width ⟨0,by decide⟩ (fun j=>bounded (fun k=>lower p runtime (xs k)) j.val)) zero
  | .add (.add .zero (.row n xs)) b=>.add32 (lower p runtime b)
      (.add32 (rowSum p.width ⟨0,by decide⟩ (fun j=>bounded (fun k=>lower p runtime (xs k)) j.val)) zero)
  | .add a b=>.add32 (lower p runtime a) (lower p runtime b)
  | .sub (.mul a b) (.mul c d)=>
      if p.reverseRowOperands then .fma32 (lower p runtime b) (lower p runtime a) (.neg32 (.mul32 (lower p runtime d) (lower p runtime c)))
      else .fma32 (lower p runtime a) (lower p runtime b) (.neg32 (.mul32 (lower p runtime c) (lower p runtime d)))
  | .sub a b=>.sub32 (lower p runtime a) (lower p runtime b)
  | .mul a .zero=>.mul32 (.bits runtime.scaleBits) (lower p runtime a)
  | .mul a (.literal _)=>.mul32 (.bits runtime.scaleBits) (lower p runtime a)
  | .mul a b=>.mul32 (lower p runtime a) (lower p runtime b)
  | .minus a=>.sub32 zero (lower p runtime a)
  | .exp a=>.exp2Approx Bool.false (.mul32 (lower p runtime a) (.bits 0x3fb8aa3b))
  | .dot D r c x y=>dotTF32 D r c (fun j=>lower p runtime (x j)) (fun j=>lower p runtime (y j)) zero
  | .total _ xs=>p.historySum (bounded (fun j=>lower p runtime (xs j)))
  | .row _ xs=>rowSum p.width ⟨0,by decide⟩ (fun j=>bounded (fun k=>lower p runtime (xs k)) j.val)

def literal (x : ℝ) : Stage := if x=0 then .zero else .literal x

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Stage .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : Stage :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i j)

def sum (_ : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Stage .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep)) : Stage :=
  if shape.length=1 then .total _ (reductionInputs ax keep xs i)
  else .row _ (reductionInputs ax keep xs i)

def algebra (p : Descriptor) (runtime : Runtime) : Algebra Stage where
  literal:=fun _ dtype x=>if dtype=.fp32 then .ready (if x=0 then zero else .literal32 x) else literal x
  negInf:=.ready (.bits 0xff800000)
  binary:=fun _ _ op a b=>match op with
    | .add=>.add a b | .sub=>.sub a b | .mul=>.mul a b | _=>.zero
  unary:=fun _ op a=>match op with
    | .exp=>.exp a | .neg _=>.minus a | _=>.zero
  cast:=fun _ _ _ a=>a
  fromNat:=fun _ n=>literal n
  fromInt:=fun _ n=>literal n
  fp32Bits:=fun b=>.ready (.bits b.bits)
  fp32Load:=fun a=>.ready (lower p runtime a)
  reduceSum:=sum
  reduceMax:=sum

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Stage left.toTileDType (batch++[m,k]) →
      Values Stage right.toTileDType (batch++[k,n]) →
      Values Stage out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (.dot D i.1.val i.2.1.val
      (fun d=>toFloat left (a (i.1,d,()))) (fun d=>toFloat right (b (d,i.2.1,()))))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds
      (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel : Typed.Dot Stage where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues p l r out options batch a b

def referenceQ (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Stage)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Graph :=
  lower p runtime (Typed.expectedQ (algebra p runtime) dotModel c s h t e)
def referenceK (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Stage)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Graph :=
  lower p runtime (Typed.expectedK (algebra p runtime) dotModel c s h t e)
def referenceG (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Stage)
    (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : Graph :=
  lower p runtime (Typed.expectedG (algebra p runtime) dotModel c s k h t)

theorem source_native (p : Descriptor) (runtime : Runtime) (c : Config) (s : State Stage) :
    (Typed.finalState (algebra p runtime) dotModel c s).mem=
      (Typed.written (algebra p runtime) dotModel c s
        (Typed.accumulated (algebra p runtime) dotModel c s (Typed.valueBlocks c))).mem :=
  Typed.source_memory (algebra p runtime) dotModel c s

theorem negative_zero_scale_preserved (p : Descriptor) (a : Stage) :
    lower p ⟨0x80000000⟩ (.mul a .zero)=
      .mul32 (.bits 0x80000000) (lower p ⟨0x80000000⟩ a) := rfl

#axiomsClean negative_zero_scale_preserved
#specNonCircular referenceQ avoiding [Typed.runGrid,Typed.invoke,FP.Structural.exec]
#specNonCircular referenceK avoiding [Typed.runGrid,Typed.invoke,FP.Structural.exec]
#specNonCircular referenceG avoiding [Typed.runGrid,Typed.invoke,FP.Structural.exec]
#axiomsClean source_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
