import bench.repairs.tritonbench_g.max_reduction.Proofs.Execution
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.MaxReduction.Source
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Reduction implementations are explicit here; the native instance below
uses finite trees, including the paired value/index comparison schedule. -/
structure Engine (A : Type) where
  padding : A
  maximum : (width : Nat) → (Fin width → A) → A
  indexed : (width : Nat) → (Fin width → A) → A × Nat

def reduceWith (operation : (width : Nat) → (Fin width → A) → A)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values A .real shape) :
    Values A .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.insertAxisIndex shape ax out i))
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.replaceAxisIndex shape ax out i))

def indexedWith (operation : (width : Nat) → (Fin width → A) → A × Nat)
    (dtype : FloatDType) {shape : TileShape} (ax : Fin shape.length)
    (xs : Values A dtype.toTileDType shape) :
    Option (Values A dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values A .nat (TileShape.eraseAxis shape ax)) :=
  let result := fun out => operation (TileShape.axisDim shape ax)
    (fun i => toFloat dtype (xs (TileShape.insertAxisIndex shape ax out i)))
  some (fun out => ofFloat dtype (result out).1,fun out => (result out).2)

def Engine.algebra (e : Engine A) : Algebra A where
  literal := fun _ _ _ => e.padding
  negInf := e.padding
  binary := fun _ _ _ _ _ => e.padding
  unary := fun _ _ _ => e.padding
  cast := fun _ _ _ => id
  fromNat := fun _ _ => e.padding
  fromInt := fun _ _ => e.padding
  fp32Bits := fun _ => e.padding
  fp32Load := id
  reduceMax := fun _ => reduceWith e.maximum
  reduceSum := fun _ => reduceWith e.maximum
  indexedMax := fun _ => indexedWith e.indexed

def scalarValue (e : Engine A) (input : Nat → A) (N B start : Nat) : A :=
  e.maximum B (fun i => if start+i.val<N then input (start+i.val) else e.padding)

theorem stage1_exec [Inhabited A] (e : Engine A) (inp mid : RegionName)
    (N B : Nat) (s : State A) (hb : 0<B) :
    ∃ final,FP.Structural.exec e.algebra (stage1 inp mid N B) s=some final ∧
      final.mem=(s.write mid (s.pids 0) (.mk .real
        (scalarValue e (fun i => (s.mem inp i).read .real) N B (s.pids 0*B)))).mem := by
  simp [stage1,VeriTile.Bench.TritonBenchG.MaxReduction.stage1,FP.Structural.exec,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,
    reduceWith,scalarValue,TileShape.allIndices,TileShape.axisDim,hb,State.write,State.setReg]
  simp only [← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt, Int.toNat_natCast]
  rfl

theorem stage2_exec [Inhabited A] (e : Engine A) (mid out : RegionName)
    (N B : Nat) (s : State A) (hb : 0<B) :
    ∃ final,FP.Structural.exec e.algebra (stage2 mid out N B) s=some final ∧
      final.mem=(s.write out 0 (.mk .real
        (scalarValue e (fun i => (s.mem mid i).read .real) N B 0))).mem := by
  simp [stage2,VeriTile.Bench.TritonBenchG.MaxReduction.stage2,FP.Structural.exec,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,
    reduceWith,scalarValue,TileShape.allIndices,TileShape.axisDim,hb,State.write,State.setReg]
  rfl

def rowValue (e : Engine A) (input : Nat → A) (M N K BM BN pid k lane : Nat) : A × Nat :=
  e.indexed BN (fun n => if pid*BM+lane<M ∧ n.val<N then
    input ((pid*BM+lane)*N*K+n.val*K+k) else e.padding)

def dimensionStores [Inhabited A] (e : Engine A) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (s : State A) : State A :=
  let values := rowValue e (fun i => (s.mem inp i).read .real) M N K BM BN (s.pids 0) (s.pids 1)
  let withValues := (TileShape.allIndices [BM]).foldl (fun state i =>
    if s.pids 0*BM+i.1.val<M then state.write ov ((s.pids 0*BM+i.1.val)*K+s.pids 1)
      (.mk .real (values i.1.val).1) else state) s
  (TileShape.allIndices [BM]).foldl (fun state i =>
    if s.pids 0*BM+i.1.val<M then state.write oi ((s.pids 0*BM+i.1.val)*K+s.pids 1)
      (.mk .int ((values i.1.val).2 : Int)) else state) withValues

@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell A) (active : ι → Prop) [DecidablePred active]
    (events : List ι) (s : State A) :
    (events.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).regs=s.regs := by
  rw [State.masked_scatter_eq_filter,State.scatter_regs]

@[simp] theorem masked_mem {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell A) (active : ι → Prop) [DecidablePred active]
    (events : List ι) (s : State A) :
    (events.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).mem=
      events.foldl (fun mem i => if active i then
        fun r' o' => if r'=r ∧ o'=off i then val i else mem r' o' else mem) s.mem := by
  induction events generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split_ifs <;> rw [ih] <;> rfl

theorem dimension_exec [Inhabited A] (e : Engine A) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (s : State A) (hn : 0<BN) :
    ∃ final,FP.Structural.exec e.algebra (dimension inp ov oi M N K BM BN) s=some final ∧
      final.mem=(dimensionStores e inp ov oi M N K BM BN s).mem := by
  simp [dimension,VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource,FP.Structural.exec,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,
    indexedWith,rowValue,dimensionStores,TileShape.allIndices,TileShape.axisDim,TileShape.insertAxisIndex,toFloat,ofFloat,hn,Nat.ne_of_gt hn,
    State.setReg]
  simp only [← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt, Int.toNat_natCast]

/-- Exact native/Real finite-tree construction; invalid fallback leaves never
occur in a ReductionPlan. Thus whole-reduction equality is proved, not assumed. -/
def treeEngine (algebra : Compute.Algebra A) (plans : Nat → Tree) : Engine A where
  padding := algebra.padding
  maximum := fun width values => (plans width).eval algebra.maximum
    (fun i => if hi : i<width then values ⟨i,hi⟩ else algebra.padding)
  indexed := fun width values => (plans width).eval algebra.indexedCombine
    (fun i => (if hi : i<width then values ⟨i,hi⟩ else algebra.padding,i))

theorem scalar_tree (algebra : Compute.Algebra A) (plans : Nat → Tree)
    (B : Nat) (complete : ∀ i,i∈(plans B).leaves → i<B)
    (input : Nat → A) (N start : Nat) :
    scalarValue (treeEngine algebra plans) input N B start=
      Compute.scalar algebra (plans B) input N start := by
  apply Compute.tree_congr
  intro i hi
  simp [complete i hi,treeEngine]

theorem row_tree (algebra : Compute.Algebra A) (plans : Nat → Tree)
    (BN : Nat) (complete : ∀ i,i∈(plans BN).leaves → i<BN)
    (input : Nat → A) (M N K BM pid k lane : Nat) (active : pid*BM+lane<M) :
    rowValue (treeEngine algebra plans) input M N K BM BN pid k lane=
      (plans BN).eval algebra.indexedCombine
        (fun n => (if n<N then input ((pid*BM+lane)*N*K+n*K+k) else algebra.padding,n)) := by
  apply Compute.tree_congr
  intro i hi
  simp [complete i hi,active,treeEngine]

/-- Dtype erasure retains a supported Algorithm projection of the actual raw
source, including its widened integer address operations. -/
theorem stage1_algorithm (inp out : RegionName) (N B : Nat) :
    (stage1 inp out N B).eraseDType.toAlgorithm? =
      .ok (stage1 inp out N B).eraseDType.toAlgKernel := by
  have h : (stage1 inp out N B).toAlgorithm? = .ok (stage1 inp out N B).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]
theorem stage2_algorithm (inp out : RegionName) (N B : Nat) :
    (stage2 inp out N B).eraseDType.toAlgorithm? =
      .ok (stage2 inp out N B).eraseDType.toAlgKernel := by
  have h : (stage2 inp out N B).toAlgorithm? = .ok (stage2 inp out N B).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]
#axiomsClean stage1_algorithm
#axiomsClean stage2_algorithm

#axiomsClean stage1_exec
#axiomsClean stage2_exec
#axiomsClean dimension_exec
#axiomsClean scalar_tree
#axiomsClean row_tree
end
end VeriTile.Bench.Repairs.MaxReduction.Source
