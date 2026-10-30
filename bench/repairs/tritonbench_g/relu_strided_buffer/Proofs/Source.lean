import bench.repairs.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralTile

/-! Source execution bridge. The readable DSL retains signed pointer-register
values; the established signed tile uses natural cell-index registers. Those
private registers differ. The raw helper retains its target-format zero literal;
the historical inserted cast is used only through an explicit proof adapter.
Signed address conversion follows the complete signed sum. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def parameters (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,a.B,a.T,a.oneTile⟩

def views (c : Parameters) : SignedViews :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B⟩

/-- Extracted from the actual one-tile source, not a second handwritten AST. -/
def perTile (c : Parameters) : List ComputeStmt :=
  match ((oneTileKernel c).surfaceBody[2]? : Option ComputeStmt) with
  | some (.ifThenElse _ yes _) => yes.drop 1
  | _ => []

/-- The raw helper's weak integer zero is a target-format literal. The historical
hand translation inserted a real-to-FP32 cast instead. This proof-only adapter
lets us reuse its generic memory/frame lemmas without assuming those two
floating events agree for an arbitrary algebra. Executable kernels use M. -/
def literalBridge (M : Algebra α) : Algebra α :=
  { M with cast := fun precision source target value =>
      if source = .real ∧ target = .fp32 then M.literal none .fp32 0
      else M.cast precision source target value }

def scalarZero (M : Algebra α) : α := M.literal none .fp32 0

def scalarResult (M : Algebra α) (cmp : α → α → Bool) (raw : α) : α :=
  let input := M.cast none .fp32 .fp32 raw
  M.cast none .fp32 .fp32 (if cmp (scalarZero M) input then input else scalarZero M)

def selectedValues [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (s : State α) : Values α .fp32 [c.B] :=
  fun i => if cmp (scalarZero M) (StructuralTile.inputValues M c T s i)
    then StructuralTile.inputValues M c T s i else scalarZero M

def tileOutput [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (mem : StructuralTile.Memory α) : StructuralTile.Memory α :=
  StructuralTile.tileOutput (literalBridge M) cmp c T mem

theorem literalBridge_result (M : Algebra α) (cmp : α → α → Bool) (raw : α) :
    StructuralTile.scalarResult (literalBridge M) cmp raw = scalarResult M cmp raw := rfl

/-- Concrete profiles may establish this local equality; the generic source
execution and candidate equivalence never assume a floating cast identity. -/
theorem scalarResult_eq_historical (M : Algebra α) (cmp : α → α → Bool)
    (zero : scalarZero M = StructuralTile.scalarZero M) (raw : α) :
    scalarResult M cmp raw = StructuralTile.scalarResult M cmp raw := by
  simp only [scalarResult,StructuralTile.scalarResult,zero]

def beforeStore [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => T * c.B)
  let s := s.setReg "__veritile_block_in0_bptr_coordinates" .int [c.B]
    (fun i => ((T * c.B : Nat) : Int) + (i.1.val : Int))
  let s := s.setReg "__veritile_block_in0_bptr_shape" .int [] (fun _ => (c.n : Int))
  let s := s.setReg "in0_bptr" .int [c.B]
    (fun i => c.input.base + ((T * c.B + i.1.val : Nat) : Int) * c.input.stride)
  let s := s.setReg "in0" .fp32 [c.B] (inputValues M (views c) T s)
  let s := s.setReg "out0" .fp32 [c.B] (selectedValues M cmp (views c) T s)
  let s := s.setReg "__veritile_block_out0_bptr_coordinates" .int [c.B]
    (fun i => ((T * c.B : Nat) : Int) + (i.1.val : Int))
  let s := s.setReg "__veritile_block_out0_bptr_shape" .int [] (fun _ => (c.n : Int))
  s.setReg "out0_bptr" .int [c.B]
    (fun i => c.output.base + ((T * c.B + i.1.val : Nat) : Int) * c.output.stride)

def sourceFinal [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.output.region ((views c).outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M cmp (rawValues (views c) T s i)))
    else current) (beforeStore M cmp c T s)

theorem perTile_exact [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (perTile c) s = some (sourceFinal M cmp c T s) := by
  simp [perTile,oneTileKernel,originalBody,ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,
    evalOp_unfold,numeric,numericLt,numericLe,natLt,natLe,FP.Structural.bop,store,hc,ht,
    sourceFinal,beforeStore,views,rawValues,StructuralTile.inputValues,selectedValues,
    FloatDType.comparableDType,
    scalarResult,scalarZero,Int.natCast_add,Int.natCast_mul,View.address,View.signedAddress,Region.cast,ofFloat,toFloat]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.natCast_nonneg,
    decide_true,Bool.true_and,true_and,Nat.cast_lt]
  norm_num only
  congr 12 <;> (try funext i) <;>
    simp [FP.Structural.bop,StructuralTile.inputValues,selectedValues,rawValues,scalarZero,
      views,View.address,View.signedAddress,Int.natCast_add,Int.natCast_mul]

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : Parameters) (T : Nat) (s : State α) :
    (sourceFinal M cmp c T s).mem = tileOutput M cmp (views c) T s.mem := by
  unfold sourceFinal tileOutput StructuralTile.tileOutput
  rw [scatter_mem]
  change _ = List.foldl _ s.mem _
  apply List.foldl_ext
  intro mem i _
  by_cases hi : T * c.B + i.1.val < c.n <;> simp [hi,rawValues,views,literalBridge_result]

theorem tile_implementation [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) :
    StructuralControl.TileImplementation M (tileOutput M cmp (views c)) (perTile c) := by
  intro T s ht
  have control := scatter_control c.output.region
    (fun i : TileIndex [c.B] => (views c).outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M cmp (rawValues (views c) T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M cmp c T s)
  refine ⟨sourceFinal M cmp c T s,perTile_exact M cmp hc c T s ht,
    ⟨sourceFinal_mem M cmp c T s,control.2.1⟩,?_,?_⟩
  · rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]

#axiomsClean literalBridge_result
#axiomsClean scalarResult_eq_historical
#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
