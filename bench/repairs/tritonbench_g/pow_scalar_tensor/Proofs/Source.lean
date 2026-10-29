import bench.repairs.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeTile

/-! Execute the readable signed source using the established opaque-power tile
semantics. Pointer temporaries keep signed offsets until the actual access.
Only those private registers differ from the historical cell-index lowering. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def parameters (base : ℝ) (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,base,a.B,a.T,a.oneTile⟩

def config (c : Parameters) (candidate : Bool) : WholeSource.Config :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B,c.base,candidate⟩

def sourceKernel (c : Parameters) (candidate : Bool) : ComputeKernel :=
  if candidate then commutedKernel c else kernel c

/-- The original `.to(in0_ptr.type.element_ty)` is an observable FP32-to-FP32
cast before the power operand conversion. The historical flattened source
omitted it. This proof adapter retains that event without assuming it is an
identity for an arbitrary floating interpretation. -/
def sourceCastBridge (M : Algebra α) : Algebra α :=
  { M with cast := fun precision source target value =>
      if precision = some .fp32 ∧ source = .fp32 ∧ target = .real then
        M.cast precision source target (M.cast none .fp32 .fp32 value)
      else M.cast precision source target value }

def scalarResult (M : Algebra α) (base : ℝ) (raw : α) : α :=
  WholeTile.scalarResult (sourceCastBridge M) base raw

def inputValues [Inhabited α] (M : Algebra α) (c : WholeSource.Config)
    (T : Nat) (s : State α) : Values α .fp32 [c.B] :=
  fun i => M.cast none .fp32 .fp32 (rawValues c T s i)

def selectedValues [Inhabited α] (M : Algebra α) (c : WholeSource.Config)
    (T : Nat) (s : State α) : Values α .real [c.B] :=
  fun i => M.binary (some .fp32) .real .pow (scalarBase M c.base)
    (M.cast (some .fp32) .fp32 .real (inputValues M c T s i))

def tileOutput [Inhabited α] (M : Algebra α) (c : WholeSource.Config)
    (T : Nat) (mem : WholeTile.Memory α) : WholeTile.Memory α :=
  WholeTile.tileOutput (sourceCastBridge M) c T mem

theorem bridge_scalar_base (M : Algebra α) (base : ℝ) :
    scalarBase (sourceCastBridge M) base = scalarBase M base := rfl

/-- Tile extraction from the canonical source, not a second handwritten AST. -/
def perTile (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  match (((if candidate then oneTileCommuted c else oneTileKernel c).surfaceBody)[2]? : Option ComputeStmt) with
  | some (.ifThenElse _ yes _) => yes.drop 1
  | _ => []

def beforeStore [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => if candidate then c.B * T else T * c.B)
  let s := s.setReg "__veritile_block_in0_bptr_coordinates" .int [c.B]
    (fun i => ((T * c.B : Nat) : Int) + (i.1.val : Int))
  let s := s.setReg "__veritile_block_in0_bptr_shape" .int [] (fun _ => (c.n : Int))
  let s := s.setReg "in0_bptr" .int [c.B]
    (fun i => c.input.base + ((T * c.B + i.1.val : Nat) : Int) * c.input.stride)
  let s := s.setReg "in0" .fp32 [c.B] (inputValues M (config c candidate) T s)
  let s := s.setReg "out0" .real [c.B] (selectedValues M (config c candidate) T s)
  let s := s.setReg "__veritile_block_out0_bptr_coordinates" .int [c.B]
    (fun i => ((T * c.B : Nat) : Int) + (i.1.val : Int))
  let s := s.setReg "__veritile_block_out0_bptr_shape" .int [] (fun _ => (c.n : Int))
  s.setReg "out0_bptr" .int [c.B]
    (fun i => c.output.base + ((T * c.B + i.1.val : Nat) : Int) * c.output.stride)

def sourceFinal [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.output.region ((config c candidate).outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M c.base (rawValues (config c candidate) T s i)))
    else current) (beforeStore M c candidate T s)

theorem perTile_exact [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (perTile c candidate) s = some (sourceFinal M c candidate T s) := by
  cases candidate <;>
    simp [perTile,oneTileKernel,oneTileCommuted,originalBody,commutedBody,ComputeKernel.surfaceBody,run,step,stepAlg,
      evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,FP.Structural.bop,store,ht,
      sourceFinal,beforeStore,config,rawValues,inputValues,selectedValues,WholeTile.scalarResult,sourceCastBridge,
      scalarResult,scalarBase,SignedView.View.address,SignedView.View.signedAddress,
      Region.cast,ofFloat,toFloat,Nat.mul_comm,numericLt,numericLe,natLe,
      FloatDType.comparableDType,Int.natCast_add,Int.natCast_mul]
  all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.natCast_nonneg,
    decide_true,Bool.true_and,true_and,Nat.cast_lt]
  all_goals norm_num only
  all_goals congr 12 <;> (try funext i) <;>
    simp [FP.Structural.bop,inputValues,selectedValues,rawValues,config,
      scalarBase,Nat.mul_comm,Int.natCast_add,Int.natCast_mul,SignedView.View.address,SignedView.View.signedAddress]

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) :
    (sourceFinal M c candidate T s).mem = tileOutput M (config c candidate) T s.mem := by
  unfold sourceFinal tileOutput WholeTile.tileOutput
  rw [scatter_mem]
  change _ = List.foldl _ s.mem _
  apply List.foldl_ext
  intro mem i _
  by_cases hi : T * c.B + i.1.val < c.n <;> simp [hi,rawValues,config,scalarResult]

theorem tile_implementation [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool) :
    StructuralControl.TileImplementation M (tileOutput M (config c candidate)) (perTile c candidate) := by
  intro T s ht
  have control := scatter_control c.output.region
    (fun i : TileIndex [c.B] => (config c candidate).outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M c.base (rawValues (config c candidate) T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M c candidate T s)
  refine ⟨sourceFinal M c candidate T s,perTile_exact M c candidate T s ht,
    ⟨sourceFinal_mem M c candidate T s,control.2.1⟩,?_,?_⟩
  · rw [show (sourceFinal M c candidate T s).regs = (beforeStore M c candidate T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M c candidate T s).regs = (beforeStore M c candidate T s).regs from control.1]
    simp [beforeStore]

#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
