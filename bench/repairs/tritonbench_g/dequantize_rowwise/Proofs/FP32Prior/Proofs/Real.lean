import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Direct
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Real
open VeriTile.Triton FP.Structural Bytes Direct
noncomputable section

def logical (m : Nat → ℝ) (t : Tensor) (i : Nat) : ℝ :=
  if t.metadata.isNeg then -m (t.storage.address (t.physicalIndex i))
  else m (t.storage.address (t.physicalIndex i))

def target (c : Request) (m : Nat → ℝ) (q : Coordinates c) : ℝ :=
  logical m c.scale q.1.val * logical m c.x (linear q) / 127

def backend (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with
    fp32Load:=id
    cast:=fun _ _ _ x=>x
    literal:=fun _ _ x=>x
    binary:=fun precision dtype op x y=>match op with
      | .mul=>x*y | _=>rest.binary precision dtype op x y}

theorem scalar (rest : Algebra ℝ) (scale x : ℝ) :
    Execution.scalar (backend rest) scale x=scale*x/127 := by
  simp [Execution.scalar,backend,div_eq_mul_inv]

def initial (c : Request) (out : Allocation c) (m : Nat → ℝ) : State ℝ where
  mem := fun region index=>if region=xRegion then .mk .real (m ((xView c).base+4*index))
    else if region=scaleRegion then .mk .real (m ((scaleView c).base+4*index))
    else .mk .fp16 (m (out.base+2*index))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun dim=>if dim=0 then rows c.x.metadata else 1
  undef := fun d _ _=>defaultValue d

def order (c : Request) : List Nat := if c.x.metadata.numel=0 then [] else List.range (rows c.x.metadata)
structure Returned where
  object : Nat
  metadata : TensorMetadata
  state : State ℝ

def run (rest : Algebra ℝ) (c : Request) (out : Allocation c) (m : Nat → ℝ) : Except InputError (Option Returned) := do
  checkTensorInputs c.x.metadata c.scale.metadata
  return (Grid.execute (backend rest) (p c) (order c) (initial c out m)).map (fun t=>⟨out.object,out.metadata,t⟩)

def Result (c : Request) (out : Allocation c) (m : Nat → ℝ) (ret : Returned) : Prop :=
  ret.object=out.object ∧ ret.metadata=out.metadata ∧ ret.metadata.shape=c.x.metadata.shape ∧
  ret.metadata.dtype=.other "float16" ∧ ret.metadata.device=c.x.metadata.device ∧
  (∀ q : Coordinates c,ret.state.mem outRegion (linear q)=.mk .fp16 (target c m q)) ∧
  (∀ region i,(region≠outRegion ∨ ∀ q : Coordinates c,i≠linear q) →
    ret.state.mem region i=(initial c out m).mem region i)

theorem direct_logical (m : Nat → ℝ) (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (noNeg : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    m ((VeriTile.Torch.FP32.dataView t laws contiguous).base+4*i)=logical m t i := by
  simp only [logical,noNeg,Bool.false_eq_true,↓reduceIte,laws.contiguous contiguous i hi]
  congr 1
  simp only [VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,
    if_neg (by omega : t.metadata.numel≠0),VeriTile.Torch.FP32.Storage.address]
  omega

theorem value_target (rest : Algebra ℝ) (c : Request) (out : Allocation c) (m : Nat → ℝ) (q : Coordinates c) :
    Grid.value (backend rest) (p c) (initial c out m) q.1.val q.2.val=target c m q := by
  have xi : linear q<c.x.metadata.numel := by simpa [numel_eq _ _ c.contract] using linear_bound q
  have si : q.1.val<c.scale.metadata.numel := lt_of_lt_of_le q.1.isLt (c.contract.scales (nonempty_coordinate c q))
  simp only [Grid.value,initial,xRegion,scaleRegion,Region.mk.injEq,String.reduceEq,↓reduceIte,Cell.read_same,scalar]
  change m ((scaleView c).base+4*q.1.val)*m ((xView c).base+4*linear q)/127=_
  unfold xView scaleView
  rw [direct_logical m c.x c.xLaws c.contract.contiguous.1 c.contract.noNeg.1 _ xi,
    direct_logical m c.scale c.scaleLaws c.contract.contiguous.2 c.contract.noNeg.2 _ si]
  rfl

theorem whole (rest : Algebra ℝ) (c : Request) (out : Allocation c) (m : Nat → ℝ) :
    ∃ ret,run rest c out m=.ok (some ret) ∧ Result c out m ret := by
  obtain ⟨t,ht,read,frame⟩:=Grid.order_correct (backend rest) (p c) (order c) (initial c out m)
  refine ⟨⟨out.object,out.metadata,t⟩,?_,rfl,rfl,out.shape,out.dtype,out.device,?_,?_⟩
  · simp [run,(checkTensorInputs_iff _ _).mpr c.contract,ht]
  · intro q
    have member : q.1.val∈order c := by
      simp [order,Nat.ne_of_gt (nonempty_coordinate c q),q.1.isLt]
    have value:=read q.1.val member q.2.val q.2.isLt
    rw [value_target rest c out m q] at value
    exact value
  · intro region i miss
    apply frame
    rcases miss with h|h
    · exact Or.inl h
    · right
      intro r hr d hd
      have rb : r<rows c.x.metadata := by
        by_cases empty : c.x.metadata.numel=0
        · simp [order,empty] at hr
        · simpa [order,empty] using hr
      exact h (⟨r,rb⟩,⟨d,hd⟩)

#specNonCircular target avoiding [run,Grid.execute,kernel,source,Execution.scalar]
#axiomsClean whole
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Real
