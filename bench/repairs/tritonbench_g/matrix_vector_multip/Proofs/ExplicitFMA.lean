import bench.repairs.tritonbench_g.matrix_vector_multip.CandidateIO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeSource WholeNative
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def codePrefix (N M BN BM : Nat) := (Kernels.explicitFMA N M BN BM).surfaceBody.take 7
def body (N M BN BM : Nat) : List ComputeStmt :=
  match ((Kernels.explicitFMA N M BN BM).surfaceBody[7]? : Option ComputeStmt) with
  | some (.forRange _ _ _ _ code) => code
  | _ => []
def suffix (N M BN BM : Nat) := (Kernels.explicitFMA N M BN BM).surfaceBody.drop 8

theorem decomposition (N M BN BM : Nat) :
    (Kernels.explicitFMA N M BN BM).surfaceBody = codePrefix N M BN BM ++
      [.forRange "m" 0 M BM (body N M BN BM)] ++ suffix N M BN BM := rfl

theorem prefix_runs (native : Scalars) (g : Graph BN BM) (N M : Nat) (s : State Payload) :
    run (algebra native g) (codePrefix N M BN BM) s =
      some (prefixState (WholeNative.algebra native g) (parameters N M BN BM) (s.pids 0) s) := by
  simp [codePrefix,Kernels.explicitFMA,parameters,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,Region.cast,
    prefixState,WholeSource.row,aOffset,bOffset,Option.bind,algebra,WholeNative.algebra]
  congr 10 <;> (first | rfl | (funext i; simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]))

theorem body_runs (native : Scalars) (g : Graph BN BM) (N M pid off : Nat)
    (memory : RegionName → Nat → Cell Payload) (acc : Acc Payload (parameters N M BN BM))
    (s : State Payload) (ctx : Context (parameters N M BN BM) pid off memory acc s) :
    run (algebra native g) (body N M BN BM) (s.setReg "m" .nat [] (fun _ => off)) =
      some (bodyState (WholeNative.algebra native g) (parameters N M BN BM) pid off memory acc s) := by
  rcases ctx with ⟨hn,hm,hmask,ha,hb,hacc,hmem⟩
  simp only [parameters] at hn hm hmask ha hb hacc
  simp [body,Kernels.explicitFMA,parameters,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,Region.cast,
    Option.bind,hn,hm,hmask,ha,hb,hacc,hmem,bodyState,nextAcc,aChunk,bChunk,load,
    algebra,WholeNative.algebra,ofFloat,toFloat,ComputeDType.eraseDType]
  congr 12 <;> (first | rfl | (funext ⟨i,j,k⟩; cases k; norm_num [aOffset,bOffset,WholeSource.row,nextAcc,aChunk,bChunk,load,bop,Broadcast.leftIndex,Broadcast.rightIndex]; try ring))

theorem range_equal (native : Scalars) (g : Graph BN BM) (N M pid off : Nat)
    (memory : RegionName → Nat → Cell Payload) (acc : Acc Payload (parameters N M BN BM))
    (s : State Payload) (ctx : Context (parameters N M BN BM) pid off memory acc s) :
    FP.Structural.range (algebra native g) "m" off M BM (body N M BN BM) s =
      FP.Structural.range (WholeNative.algebra native g) "m" off M BM
        (WholeSource.body (parameters N M BN BM)) s := by
  rw [FP.Structural.range,FP.Structural.range]
  by_cases width : BM=0
  · simp [width]
  · simp only [width,if_false]
    by_cases active : off<M
    · simp only [active,if_true]
      rw [body_runs native g N M pid off memory acc s ctx,
        WholeSource.body_runs (WholeNative.algebra native g) (parameters N M BN BM)
          pid off memory acc s ctx]
      exact range_equal native g N M pid (off+BM) memory
        (nextAcc (WholeNative.algebra native g) (parameters N M BN BM) pid off memory acc) _
        (body_context (WholeNative.algebra native g) (parameters N M BN BM) pid off memory acc s ctx)
    · simp [active]
termination_by M-off

theorem suffix_equal (native : Scalars) (g : Graph BN BM) (N M : Nat) (s : State Payload) :
    run (algebra native g) (suffix N M BN BM) s =
      run (WholeNative.algebra native g) (WholeSource.suffix (parameters N M BN BM)) s := by
  simp [suffix,Kernels.explicitFMA,WholeSource.suffix,WholeSource.source,Typed.kernel,
    Typed.source,parameters,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,algebra,WholeNative.algebra,Option.bind,numeric,bop,ofFloat,toFloat,store,Region.cast]

theorem execution_equal (native : Scalars) (g : Graph BN BM) (N M : Nat)
    (s : State Payload) :
    exec (algebra native g) (Kernels.explicitFMA N M BN BM) s =
      exec (WholeNative.algebra native g) (Kernels.source N M BN BM) s := by
  change run (algebra native g) (Kernels.explicitFMA N M BN BM).surfaceBody s =
    run (WholeNative.algebra native g) (Kernels.source N M BN BM).surfaceBody s
  rw [decomposition,KernelSource.source_binding,complete_source]
  simp only [run_append,prefix_runs,WholeSource.prefix_runs,Option.bind_some]
  simp only [run,step]
  rw [range_equal native g N M (s.pids 0) 0 s.mem _ _
    (prefix_context (WholeNative.algebra native g) (parameters N M BN BM) s)]
  congr 1
  funext final
  exact suffix_equal native g N M final

theorem prefix_safe (native : Scalars) (g : Graph BN BM) (N M : Nat)
    (bounds : RegionName → Nat) (s : State Payload) :
    Safety.traceSafeList (algebra native g) bounds (codePrefix N M BN BM) s := by
  simp [codePrefix,Kernels.explicitFMA,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,numericLt,natLt,bop,Region.cast,Option.bind]

theorem body_safe (native : Scalars) (g : Graph BN BM) (N M pid off : Nat)
    (memory : RegionName → Nat → Cell Payload) (acc : Acc Payload (parameters N M BN BM))
    (s : State Payload) (ctx : Context (parameters N M BN BM) pid off memory acc s)
    (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B") :
    Safety.traceSafeList (algebra native g) bounds (body N M BN BM)
      (s.setReg "m" .nat [] (fun _ => off)) := by
  rcases ctx with ⟨hn,hm,hmask,ha,hb,hacc,hmem⟩
  simp only [parameters] at hn hm hmask ha hb hacc
  simp [body,Kernels.explicitFMA,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,
    Region.cast,Option.bind,store,hn,hm,hmask,ha,hb,hacc,hmem,ofFloat,toFloat,
    ComputeDType.eraseDType,algebra,WholeNative.algebra]
  constructor
  · intro i j hi hj
    simpa [aOffset,parameters,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      matrix (WholeSource.row (parameters N M BN BM) pid i.val) (off+j.val) hi hj
  · intro j hj
    simpa [bOffset,parameters,Nat.add_mul,Nat.add_comm] using vector (off+j.val) hj

theorem range_safe (native : Scalars) (g : Graph BN BM) (N M : Nat) (width : 0<BM)
    (pid off : Nat) (memory : RegionName → Nat → Cell Payload)
    (acc : Acc Payload (parameters N M BN BM)) (s : State Payload)
    (ctx : Context (parameters N M BN BM) pid off memory acc s)
    (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B") :
    Safety.rangeSafe (algebra native g) bounds "m" off M BM (body N M BN BM) s := by
  rw [Safety.rangeSafe]
  simp only [Nat.ne_of_gt width,if_false]
  split
  · exact ⟨body_safe native g N M pid off memory acc s ctx bounds matrix vector,by
      rw [body_runs native g N M pid off memory acc s ctx]
      exact range_safe native g N M width pid (off+BM) memory
        (nextAcc (WholeNative.algebra native g) (parameters N M BN BM) pid off memory acc) _
        (body_context (WholeNative.algebra native g) (parameters N M BN BM) pid off memory acc s ctx)
        bounds matrix vector⟩
  · trivial
termination_by M-off

theorem source_safe (native : Scalars) (g : Graph BN BM) (N M : Nat) (width : 0<BM)
    (bounds : RegionName → Nat) (s : State Payload)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B")
    (output : ∀ i : Fin BN, WholeSource.row (parameters N M BN BM) (s.pids 0) i.val<N →
      WholeSource.row (parameters N M BN BM) (s.pids 0) i.val<bounds "C") :
    TraceSafe (algebra native g) bounds (Kernels.explicitFMA N M BN BM) s := by
  unfold TraceSafe
  rw [decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prefix_safe native g N M bounds s,?_⟩
  intro middle hprefix
  rw [prefix_runs] at hprefix
  cases hprefix
  rw [Safety.traceSafeList_append]
  constructor
  · simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold]
    refine ⟨range_safe native g N M width (s.pids 0) 0 s.mem _ _
      (prefix_context (WholeNative.algebra native g) (parameters N M BN BM) s)
      bounds matrix vector,?_⟩
    cases step (algebra native g) (.forRange "m" 0 M BM (body N M BN BM))
      (prefixState (WholeNative.algebra native g) (parameters N M BN BM) (s.pids 0) s) <;> trivial
  · intro after executed
    obtain ⟨last,final,ran,ctx⟩ := WholeSource.range_runs (WholeNative.algebra native g)
      (parameters N M BN BM) width (s.pids 0) 0 s.mem
      (fun _ => (WholeNative.algebra native g).literal (some .fp32) .real 0) _
      (prefix_context (WholeNative.algebra native g) (parameters N M BN BM) s)
    simp only [run,step] at executed
    rw [range_equal native g N M (s.pids 0) 0 s.mem _ _
      (prefix_context (WholeNative.algebra native g) (parameters N M BN BM) s)] at executed
    change FP.Structural.range (WholeNative.algebra native g) "m" 0 M BM
      (WholeSource.body (parameters N M BN BM))
      (prefixState (WholeNative.algebra native g) (parameters N M BN BM) (s.pids 0) s)=some final at ran
    rw [ran] at executed
    cases executed
    have sameSuffix : suffix N M BN BM=WholeSource.suffix (parameters N M BN BM) := rfl
    rw [sameSuffix]
    apply AccessSafety.suffix_safe (algebra native g) (parameters N M BN BM)
      (s.pids 0) last s.mem _ _ ctx bounds
    simpa [parameters] using output

theorem equivalent (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM)
    (width : 0<BM) :
    (originalIO native N M BN BM pid g).Equivalent (variant native N M BN BM pid g) := by
  intro vals _ bounds covered initial pins
  obtain ⟨safe,final,ran,values,frame⟩ :=
    KernelNativeResult.io_correct native N M BN BM pid g width vals trivial bounds covered initial pins
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change (invocation native N M BN BM pid g).Safe bounds initial ∧ _
    refine ⟨?_,fun _ _ => trivial⟩
    apply source_safe native g N M width bounds
    · intro i j hi hj
      exact covered.2.1 (0 : Fin 2) (⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact covered.2.1 (1 : Fin 2) (⟨j,hj⟩,PUnit.unit) trivial
    · intro i active
      have act : pid*BN+i.val<N := by
        simpa [WholeSource.row,parameters,invocation,IO.invocation,Invocation.initial,
          bindRegisters] using active
      simpa [WholeSource.row,parameters,invocation,IO.invocation,Invocation.initial,
        bindRegisters] using covered.2.2.1 (0 : Fin 1) (i,PUnit.unit) act
  · change ((invocation native N M BN BM pid g).run initial).bind (fun s => some s)=some final
    change (exec (algebra native g) (Kernels.explicitFMA N M BN BM)
      ((invocation native N M BN BM pid g).initial initial)).bind _=some final
    rw [execution_equal]
    exact ran
  · intro o j active
    rfl

#axiomsClean execution_equal
#axiomsClean source_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
