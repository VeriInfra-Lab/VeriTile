import bench.tritonbench_g.log_softmax.Kernels
import bench.tritonbench_g.log_softmax.Proofs.Graph
import bench.tritonbench_g.log_softmax.Proofs.Address
import bench.tritonbench_g.max_reduction.Proofs.Source
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

def signedClamp (a n : Nat) : Int :=
  if (a : Int) < (n : Int)-1 then a else (n : Int)-1

theorem clamp_expression (a n : Nat) :
    (if (a : Int) < (n : Int)-1 then (a : Int) else (n : Int)-1) = signedClamp a n := rfl

theorem signedClamp_eq (a n : Nat) (hn : 0<n) :
    signedClamp a n = (min a (n-1) : Nat) := by
  have cast_sub : ((n-1 : Nat) : Int) = (n : Int)-1 := Int.natCast_sub hn
  simp only [signedClamp, ← cast_sub, Int.ofNat_lt]
  split_ifs with h
  · rw [min_eq_left (Nat.le_of_lt h)]
  · rw [min_eq_right (Nat.le_of_not_gt h)]

theorem signedAddress_eq (M N K a b k : Nat) (hm : 0<M) (hn : 0<N) (hk : 0<K) :
    (signedClamp a M * N * K + signedClamp b N * K + signedClamp k K).toNat =
      min a (M-1)*N*K+min b (N-1)*K+min k (K-1) := by
  simp only [signedClamp_eq _ _ hm, signedClamp_eq _ _ hn, signedClamp_eq _ _ hk,
    -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

/-- Execution address in the source's signed index channel. Its dense
mathematical coordinate is proved separately under the actual access mask. -/
def address (M N K BM p k : Nat) (i : TileIndex [BM,C]) : Nat :=
  (signedClamp (p*BM+i.1.val) M * N * K + signedClamp i.2.1.val N * K + signedClamp k K).toNat

theorem address_nat (M N K BM p k : Nat) (i : TileIndex [BM,C])
    (hm : 0<M) (hn : 0<N) (hk : 0<K) :
    address M N K BM p k i =
      min (p*BM+i.1.val) (M-1)*N*K+min i.2.1.val (N-1)*K+min k (K-1) :=
  signedAddress_eq M N K _ _ _ hm hn hk

/-- The native signed clamp agrees with the natural mathematical address in
active lanes. Empty dimensions remain inactive, so no positivity premise is
added to the public kernel theorem. -/
theorem signed_masked_read [Inhabited A] (memory : RegionName → Nat → Cell A)
    (r : RegionName) (M N K a b k : Nat) (fallback : A) :
    (if a<M ∧ b<N ∧ k<K then
      (memory r (signedClamp a M * N * K + signedClamp b N * K + signedClamp k K).toNat).read .real
    else fallback) =
    (if a<M ∧ b<N ∧ k<K then
      (memory r (min a (M-1)*N*K+min b (N-1)*K+min k (K-1))).read .real
    else fallback) := by
  by_cases h : a<M ∧ b<N ∧ k<K
  · simp only [if_pos h, signedAddress_eq M N K a b k (by omega) (by omega) (by omega)]
  · simp only [if_neg h]

theorem signed_masked_write (memory : RegionName → Nat → Cell A)
    (r : RegionName) (M N K a b k : Nat) (value : Cell A) :
    (if a<M ∧ b<N ∧ k<K then
      (fun r' o' => if r'=r ∧ o'=(signedClamp a M * N * K + signedClamp b N * K + signedClamp k K).toNat
        then value else memory r' o')
    else memory) =
    (if a<M ∧ b<N ∧ k<K then
      (fun r' o' => if r'=r ∧ o'=min a (M-1)*N*K+min b (N-1)*K+min k (K-1)
        then value else memory r' o')
    else memory) := by
  by_cases h : a<M ∧ b<N ∧ k<K
  · simp only [if_pos h, signedAddress_eq M N K a b k (by omega) (by omega) (by omega)]
  · simp only [if_neg h]

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
    ∃ final,FP.Structural.exec algebra (Source.forwardKernel O X M N K BM C) s=some final := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.forwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,hc,Nat.ne_of_gt hc,
    State.setReg]

theorem backward_total [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) :
    ∃ final,FP.Structural.exec algebra (Source.backwardKernel Y G O M N K BM C) s=some final := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.backwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg]

theorem forward_exec [Inhabited A] (e : Engine A C) (O X : RegionName)
    (M N K BM : Nat) (s : State A) (hc : 0<C) :
    ∃ final,FP.Structural.exec e.algebra (Source.forwardKernel O X M N K BM C) s=some final ∧
      final.mem=(stores O M N K BM C s (forwardValue e X M N K BM s)).mem := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.forwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,reduceWith,extend,
    active,address,signedClamp,loaded,forwardValue,stores,TileShape.axisDim,TileShape.insertAxisIndex,
    TileShape.eraseAxis,TileShape.dropInsertedIndex,
    toFloat,ofFloat,hc,Nat.ne_of_gt hc,State.setReg,and_assoc,
    VeriTile.Bench.Repairs.MaxReduction.Source.masked_mem,
    -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Int.ofNat_lt,
    ]

theorem backward_exec [Inhabited A] (e : Engine A C) (Y G O : RegionName)
    (M N K BM : Nat) (s : State A) :
    ∃ final,FP.Structural.exec e.algebra (Source.backwardKernel Y G O M N K BM C) s=some final ∧
      final.mem=(stores O M N K BM C s (backwardValue e Y G M N K BM s)).mem := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.backwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,Engine.algebra,reduceWith,extend,
    active,address,signedClamp,loaded,backwardValue,stores,TileShape.axisDim,TileShape.insertAxisIndex,
    TileShape.eraseAxis,TileShape.dropInsertedIndex,
    toFloat,ofFloat,State.setReg,and_assoc,
    VeriTile.Bench.Repairs.MaxReduction.Source.masked_mem,
    -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Int.ofNat_lt,
    ]

/-- Exterior memory is preserved for arbitrary source floating semantics. -/
theorem forward_frame [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s final : State A) (hc : 0<C)
    (ran : FP.Structural.exec algebra (Source.forwardKernel O X M N K BM C) s=some final)
    (r : RegionName) (o : Nat) (outside : r≠O ∨ M*N*K≤o) :
    final.mem r o=s.mem r o := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.forwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg,hc,Nat.ne_of_gt hc,and_assoc] at ran
  subst final
  apply State.masked_scatter_frame
  rcases outside with outside | outside
  · exact Or.inl outside
  · right
    intro i _ hi same
    have active : s.pids 0*BM+i.1.val<M ∧ i.2.1.val<N ∧ s.pids 1<K := by
      simpa only [and_assoc, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt] using hi
    have bound := Address.masked_offset_bound
      (show 0<M by omega) (show 0<N by omega) (show 0<K by omega)
      (s.pids 0*BM+i.1.val) i.2.1.val (s.pids 1)
    simp only [Int.natCast_mul, Int.natCast_add, ← ite_mul] at same
    simp only [ ← Int.natCast_mul, ← Int.natCast_add, clamp_expression] at same
    change o = (signedClamp (s.pids 0*BM+i.1.val) M * N * K +
      signedClamp i.2.1.val N * K + signedClamp (s.pids 1) K).toNat at same
    rw [signedAddress_eq M N K _ _ _ (by omega) (by omega) (by omega)] at same
    simp only [Address.offset,Address.safe] at bound
    omega

/-- Exterior memory is preserved for arbitrary source floating semantics. -/
theorem backward_frame [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s final : State A)
    (ran : FP.Structural.exec algebra (Source.backwardKernel Y G O M N K BM C) s=some final)
    (r : RegionName) (o : Nat) (outside : r≠O ∨ M*N*K≤o) :
    final.mem r o=s.mem r o := by
  simp (config := { unfoldPartialApp := Bool.true }) [Source.backwardKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,TileShape.axisDim,State.setReg,and_assoc] at ran
  subst final
  apply State.masked_scatter_frame
  rcases outside with outside | outside
  · exact Or.inl outside
  · right
    intro i _ hi same
    have active : s.pids 0*BM+i.1.val<M ∧ i.2.1.val<N ∧ s.pids 1<K := by
      simpa only [and_assoc, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt] using hi
    have bound := Address.masked_offset_bound
      (show 0<M by omega) (show 0<N by omega) (show 0<K by omega)
      (s.pids 0*BM+i.1.val) i.2.1.val (s.pids 1)
    simp only [Int.natCast_mul, Int.natCast_add, ← ite_mul] at same
    simp only [ ← Int.natCast_mul, ← Int.natCast_add, clamp_expression] at same
    change o = (signedClamp (s.pids 0*BM+i.1.val) M * N * K +
      signedClamp i.2.1.val N * K + signedClamp (s.pids 1) K).toNat at same
    rw [signedAddress_eq M N K _ _ _ (by omega) (by omega) (by omega)] at same
    simp only [Address.offset,Address.safe] at bound
    omega

end
end VeriTile.Bench.Repairs.LogSoftmax.Kernel
