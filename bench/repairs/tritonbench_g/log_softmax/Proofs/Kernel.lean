import bench.repairs.tritonbench_g.log_softmax.Kernels
import bench.repairs.tritonbench_g.log_softmax.Proofs.Graph
import bench.repairs.tritonbench_g.log_softmax.Proofs.Address
import bench.repairs.tritonbench_g.max_reduction.Proofs.Source
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Successful execution of the complete canonical source, including every
load, FP32 identity cast, reduction, mask and store. The same opaque FP
primitives and reduction schedules are used by both index variants. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Kernel
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Engine (A : Type) (C : Nat) where
  ops : Ops A
  maximum : (Fin C → A) → A
  summation : (Fin C → A) → A

def extend (fallback : A) (f : (Fin C → A) → A) (width : Nat) (v : Fin width → A) : A :=
  if h : width = C then f (fun i => v (i.cast h.symm)) else fallback

def reduceWith (operation : (width : Nat) → (Fin width → A) → A)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values A .real shape) :
    Values A .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.insertAxisIndex shape ax out i))
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.replaceAxisIndex shape ax out i))

/-- The source uses only FP32 identity casts. Scalar arithmetic and exp/log are
the unchanged primitive slots, not experiments or Real arithmetic assertions. -/
def Engine.algebra (e : Engine A C) : Algebra A where
  literal := fun _ _ _ => e.ops.zero
  negInf := e.ops.negativeInfinity
  binary := fun _ _ op => match op with
    | .add => e.ops.add
    | .sub => e.ops.subtract
    | .mul => e.ops.multiply
    | .max => e.ops.maximum
    | _ => fun _ _ => e.ops.zero
  unary := fun _ op => match op with
    | .exp => e.ops.exp
    | .log => e.ops.log
    | _ => fun _ => e.ops.zero
  cast := fun _ _ _ => id
  fromNat := fun _ _ => e.ops.zero
  fromInt := fun _ _ => e.ops.zero
  fp32Bits := fun _ => e.ops.zero
  fp32Load := id
  reduceMax := fun _ => reduceWith (extend e.ops.negativeInfinity e.maximum)
  reduceSum := fun _ => reduceWith (extend e.ops.zero e.summation)

def active (M N K BM p k : Nat) (i : TileIndex [BM,C]) : Prop :=
  p*BM+i.1.val<M ∧ i.2.1.val<N ∧ k<K

def address (M N K BM p k : Nat) (i : TileIndex [BM,C]) : Nat :=
  min (p*BM+i.1.val) (M-1)*N*K+min i.2.1.val (N-1)*K+min k (K-1)

def loaded [Inhabited A] (fallback : A) (X : RegionName)
    (M N K BM C : Nat) (s : State A) : Values A .real [BM,C] :=
  fun i => if active M N K BM (s.pids 0) (s.pids 1) i then
    (s.mem X (address M N K BM (s.pids 0) (s.pids 1) i)).read .real else fallback

def forwardValue [Inhabited A] (e : Engine A C) (X : RegionName)
    (M N K BM : Nat) (s : State A) (i : TileIndex [BM,C]) : A :=
  let x := loaded e.ops.negativeInfinity X M N K BM C s
  let m := e.maximum (fun j => x (i.1,j,PUnit.unit))
  let delta := fun j => e.ops.subtract (x (i.1,j,PUnit.unit)) m
  e.ops.subtract (delta i.2.1) (e.ops.log (e.summation (fun j => e.ops.exp (delta j))))

def backwardValue [Inhabited A] (e : Engine A C) (Y G : RegionName)
    (M N K BM : Nat) (s : State A) (i : TileIndex [BM,C]) : A :=
  let y := loaded e.ops.negativeInfinity Y M N K BM C s
  let g := loaded e.ops.zero G M N K BM C s
  e.ops.subtract (g i) (e.ops.multiply (e.ops.exp (y i))
    (e.summation (fun j => g (i.1,j,PUnit.unit))))

def stores [Inhabited A] (O : RegionName) (M N K BM C : Nat)
    (s : State A) (v : TileIndex [BM,C] → A) : State A :=
  (TileShape.allIndices [BM,C]).foldl (fun t i =>
    if active M N K BM (s.pids 0) (s.pids 1) i then
      t.write O (address M N K BM (s.pids 0) (s.pids 1) i) (.mk .real (v i)) else t) s

/-- Basic source primitives are total slots. Success is constructive even for
arbitrary FP32 words, casts, special-value policies and unchanged reducers. -/
theorem forward_total [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s : State A) (hc : 0<C) :
    ∃ final,FP.Structural.exec algebra (Source.forward O X M N K BM C) s=some final := by
  simp [Source.forward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,hc,Nat.ne_of_gt hc,
    State.setReg]

theorem backward_total [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) :
    ∃ final,FP.Structural.exec algebra (Source.backward Y G O M N K BM C) s=some final := by
  simp [Source.backward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg]

theorem forward_exec [Inhabited A] (e : Engine A C) (O X : RegionName)
    (M N K BM : Nat) (s : State A) (hc : 0<C) :
    ∃ final,FP.Structural.exec e.algebra (Source.forward O X M N K BM C) s=some final ∧
      final.mem=(stores O M N K BM C s (forwardValue e X M N K BM s)).mem := by
  simp [Source.forward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,reduceWith,extend,
    active,address,loaded,forwardValue,stores,TileShape.axisDim,TileShape.insertAxisIndex,
    TileShape.eraseAxis,TileShape.dropInsertedIndex,
    toFloat,ofFloat,hc,Nat.ne_of_gt hc,State.setReg,min_def_lt,and_assoc,
    VeriTile.Bench.Repairs.MaxReduction.Source.masked_mem]

theorem backward_exec [Inhabited A] (e : Engine A C) (Y G O : RegionName)
    (M N K BM : Nat) (s : State A) :
    ∃ final,FP.Structural.exec e.algebra (Source.backward Y G O M N K BM C) s=some final ∧
      final.mem=(stores O M N K BM C s (backwardValue e Y G M N K BM s)).mem := by
  simp [Source.backward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,reduceWith,extend,
    active,address,loaded,backwardValue,stores,TileShape.axisDim,TileShape.insertAxisIndex,
    TileShape.eraseAxis,TileShape.dropInsertedIndex,
    toFloat,ofFloat,State.setReg,min_def_lt,and_assoc,
    VeriTile.Bench.Repairs.MaxReduction.Source.masked_mem]

/-- Exterior memory is preserved for arbitrary source floating semantics. -/
theorem forward_frame [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s final : State A) (hc : 0<C)
    (ran : FP.Structural.exec algebra (Source.forward O X M N K BM C) s=some final)
    (r : RegionName) (o : Nat) (outside : r≠O ∨ M*N*K≤o) :
    final.mem r o=s.mem r o := by
  simp [Source.forward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg,hc,Nat.ne_of_gt hc,and_assoc] at ran
  subst final
  apply State.masked_scatter_frame
  rcases outside with outside | outside
  · exact Or.inl outside
  · right
    intro i _ hi same
    have active : s.pids 0*BM+i.1.val<M ∧ i.2.1.val<N ∧ s.pids 1<K := by
      simpa only [and_assoc] using hi
    have bound := Address.masked_offset_bound
      (show 0<M by omega) (show 0<N by omega) (show 0<K by omega)
      (s.pids 0*BM+i.1.val) i.2.1.val (s.pids 1)
    have bound' : ((if s.pids 0*BM+i.1.val<M-1 then s.pids 0*BM+i.1.val else M-1)*N*K+
        (if i.2.1.val<N-1 then i.2.1.val else N-1)*K+
        (if s.pids 1<K-1 then s.pids 1 else K-1))<M*N*K := by
      simpa only [Address.offset,Address.safe,min_def_lt] using bound
    simp only [ite_mul] at bound'
    omega

/-- Exterior memory is preserved for arbitrary source floating semantics. -/
theorem backward_frame [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s final : State A)
    (ran : FP.Structural.exec algebra (Source.backward Y G O M N K BM C) s=some final)
    (r : RegionName) (o : Nat) (outside : r≠O ∨ M*N*K≤o) :
    final.mem r o=s.mem r o := by
  simp [Source.backward,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg,and_assoc] at ran
  subst final
  apply State.masked_scatter_frame
  rcases outside with outside | outside
  · exact Or.inl outside
  · right
    intro i _ hi same
    have active : s.pids 0*BM+i.1.val<M ∧ i.2.1.val<N ∧ s.pids 1<K := by
      simpa only [and_assoc] using hi
    have bound := Address.masked_offset_bound
      (show 0<M by omega) (show 0<N by omega) (show 0<K by omega)
      (s.pids 0*BM+i.1.val) i.2.1.val (s.pids 1)
    have bound' : ((if s.pids 0*BM+i.1.val<M-1 then s.pids 0*BM+i.1.val else M-1)*N*K+
        (if i.2.1.val<N-1 then i.2.1.val else N-1)*K+
        (if s.pids 1<K-1 then s.pids 1 else K-1))<M*N*K := by
      simpa only [Address.offset,Address.safe,min_def_lt] using bound
    simp only [ite_mul] at bound'
    omega

end
end VeriTile.Bench.Repairs.LogSoftmax.Kernel
