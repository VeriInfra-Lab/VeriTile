import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
open Safety
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure ForwardBounds (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (bounds : RegionName → Nat) : Prop where
  source : ∀ t, t<c.N-1 → ∀ (e : Fin 64) (j : Fin 16), address c p t e.val j.val<bounds "S"
  gate : ∀ t, t<c.N-1 → gateAddress c p t<bounds "D"
  initial : seed=Bool.true → ∀ (e : Fin 64) (j : Fin 16), seedAddress c p s e.val j.val<bounds "Seed"
  output : ∀ t, t<c.N → ∀ (e : Fin 64) (j : Fin 16), address c p t e.val j.val<bounds "O"

structure BackwardBounds (c : Shape) (p : Program c) (bounds : RegionName → Nat) : Prop where
  saved : ∀ t, t<c.N-1 → ∀ (e : Fin 64) (j : Fin 16), address c p t e.val j.val<bounds "S"
  gate : ∀ t, t<c.N → gateAddress c p t<bounds "D"
  incoming : ∀ t, t<c.N → ∀ (e : Fin 64) (j : Fin 16), address c p t e.val j.val<bounds "DS"
  gradient : ∀ t, t<c.N-1 → ∀ (e : Fin 64) (j : Fin 16), address c p t e.val j.val<bounds "DI"
  gateGradient : ∀ t, t<c.N-1 → gradientAddress c p t<bounds "DG"
  seedGradient : ∀ (e : Fin 64) (j : Fin 16), dseedAddress c p e.val j.val<bounds "DL"

open KernelOps in
theorem forward_block (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : ForwardBounds c s seed p bounds)
    (entry : State α) (n : Nat) (u : State α) (hn : n<c.N-1)
    (inv : KernelForwardLoop.Holds M c s seed p entry n u) :
    traceSafeList (engine M) bounds (forwardBody c) u := by
  simp (config := {maxSteps:=200000}) [forwardBody,loadScalar,loadMatrix,assign32,storeMatrix,advance,
    matrixPtr,scalarPtr,matrix,scalar,KernelOps.update,stride,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    State.setReg,FP.Structural.store,inv.sourcePtr,inv.targetPtr,inv.gatePtr,inv.acc,ptr]
  exact ⟨backed.gate n hn,backed.source n hn,backed.output (n+1) (by omega)⟩

open KernelOps in
theorem backward_block (M : Algebra α) (c : Shape) (s : SeedStrides)
    (p : Program c) (bounds : RegionName → Nat) (backed : BackwardBounds c p bounds)
    (entry : State α) (n : Nat) (u : State α) (hn : n<c.N-1)
    (inv : KernelReverseLoop.Holds M c s p entry n u) :
    traceSafeList (engine M) bounds (backwardBody c) u := by
  have sourceOff (e : Fin 64) (j : Fin 16) :=
    KernelReversePointers.live c p (c.N-2) n (by omega) e j
  have incomingOff (e : Fin 64) (j : Fin 16) :=
    KernelReversePointers.live c p (c.N-1) n (by omega) e j
  have gateOff := KernelReversePointers.gate_live c p n (by omega)
  have gradientOff := KernelReversePointers.gradient_live c p n (by omega)
  simp (config := {maxSteps:=300000}) [backwardBody,loadScalar,loadMatrix,assign32,storeMatrix,
    storeScalar,retreat,matrixPtr,scalarPtr,matrix,scalar,KernelOps.update,stride,tileStride,
    KernelOps.gradient,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    State.setReg,FP.Structural.store,inv.sourcePtr,inv.targetPtr,inv.incomingPtr,inv.gatePtr,
    inv.gradientPtr,inv.acc,inv.fixed.numK,inv.fixed.numV,ptr,
    sourceOff,incomingOff,gateOff,gradientOff]
  exact ⟨backed.saved _ (by omega),backed.incoming _ (by omega),backed.gate _ (by omega),
    backed.gateGradient _ (by omega),backed.gradient _ (by omega)⟩

theorem forward_range (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : ForwardBounds c s seed p bounds)
    (entry : State α) (n : Nat) (u : State α)
    (inv : KernelForwardLoop.Holds M c s seed p entry n u) :
    rangeSafe (engine M) bounds "_i" n (c.N-1) 1 (KernelOps.forwardBody c) u := by
  by_cases hn : n<c.N-1
  · have initialized := KernelForwardLoop.bind_holds M c s seed p entry n u inv
    obtain ⟨v,ran,held⟩ := KernelForwardLoop.iteration M c s seed p entry n _ hn initialized
    rw [rangeSafe_unfold]
    simp only [hn,ite_true,one_ne_zero,ite_false]
    refine ⟨forward_block M c s seed p bounds backed entry n _ hn initialized,?_⟩
    rw [ran]
    exact forward_range M c s seed p bounds backed entry (n+1) v held
  · rw [rangeSafe_unfold]
    simp [hn]
termination_by c.N-1-n

theorem backward_range (M : Algebra α) (c : Shape) (s : SeedStrides)
    (p : Program c) (bounds : RegionName → Nat) (backed : BackwardBounds c p bounds)
    (entry : State α) (n : Nat) (u : State α)
    (inv : KernelReverseLoop.Holds M c s p entry n u) :
    rangeSafe (engine M) bounds "_i" n (c.N-1) 1 (KernelOps.backwardBody c) u := by
  by_cases hn : n<c.N-1
  · have initialized := KernelReverseLoop.bind_holds M c s p entry n u inv
    obtain ⟨v,ran,held⟩ := KernelReverseLoop.iteration M c s p entry n _ hn initialized
    rw [rangeSafe_unfold]
    simp only [hn,ite_true,one_ne_zero,ite_false]
    refine ⟨backward_block M c s p bounds backed entry n _ hn initialized,?_⟩
    rw [ran]
    exact backward_range M c s p bounds backed entry (n+1) v held
  · rw [rangeSafe_unfold]
    simp [hn]
termination_by c.N-1-n

theorem forward_prefix (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : ForwardBounds c s seed p bounds)
    (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    traceSafeList (engine M) bounds (before .forward c s seed) u := by
  cases seed
  all_goals
    simp only [before,parts,source,Kernels.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,↓reduceIte]
    simp (config := {maxSteps:=200000}) [traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
      addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,hh,hk,hv]
  · simpa [address,layout,launch,TritonBenchG.ChunkGateRecurrence.RealProgram.address] using
      backed.output 0 nonempty
  · constructor
    · simpa [seedAddress] using backed.initial rfl
    · simpa [address,layout,launch,TritonBenchG.ChunkGateRecurrence.RealProgram.address] using
        backed.output 0 nonempty

theorem backward_prefix (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (bounds : RegionName → Nat) (u : State α) :
    traceSafeList (engine M) bounds (before .backward c s seed) u := by
  simp only [before,parts,source,Kernels.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.specializeBackward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp (config := {maxSteps:=200000}) [traceSafeList_unfold,stepSafe_unfold,
    safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
    addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]

theorem backward_post (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : BackwardBounds c p bounds)
    (carry : Lane → α) (u : State α) (nonempty : 0<c.N)
    (fixed : KernelReversePointers.Fixed c p u)
    (hx : u.regs .ptr [64,16] "DS"=some (ptr "DS" (fun e j => address c p 0 e.val j.val)))
    (hd : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gateAddress c p 0)))
    (ha : u.regs .real [64,16] "Dacc"=some carry) :
    traceSafeList (engine M) bounds (after .backward c s seed) u := by
  simp only [after,parts,source,Kernels.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.specializeBackward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp (config := {maxSteps:=200000}) [traceSafeList_unfold,stepSafe_unfold,
    safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
    addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,State.setReg,FP.Structural.store,hx,hd,ha,
    fixed.head,fixed.key,fixed.value,fixed.offsK,fixed.offsV,ptr,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  refine ⟨backed.incoming 0 nonempty,backed.gate 0 nonempty,?_⟩
  simpa only [dseedAddress] using backed.seedGradient

theorem forward_source (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : ForwardBounds c s seed p bounds)
    (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    TraceSafe (engine M) bounds (source .forward c s seed) u := by
  unfold TraceSafe
  rw [source_body,List.append_assoc,traceSafeList_append]
  refine ⟨forward_prefix M c s seed p bounds backed u nonempty hh hk hv,?_⟩
  intro v ran
  have same : v=(KernelPrefix.forward M c s seed u).val :=
    Option.some.inj (ran.symm.trans (KernelPrefix.forward M c s seed u).property)
  subst v
  have inv := KernelForwardInit.holds M c s seed p u nonempty hh hk hv
  have safe := forward_range M c s seed p bounds backed u 0 _ inv
  have tail : after .forward c s seed=[] := rfl
  simp only [tail,List.append_nil,traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · simpa [KernelProgram.loop,stepSafe_unfold,safeOp_unfold,evalOp_unfold,
      KernelOps.forward_body] using safe
  · split <;> trivial

theorem backward_source (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (bounds : RegionName → Nat) (backed : BackwardBounds c p bounds)
    (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    TraceSafe (engine M) bounds (source .backward c s seed) u := by
  unfold TraceSafe
  rw [source_body,List.append_assoc,traceSafeList_append]
  refine ⟨backward_prefix M c s seed bounds u,?_⟩
  intro v ran
  have same : v=(KernelPrefix.backward M c s seed u).val :=
    Option.some.inj (ran.symm.trans (KernelPrefix.backward M c s seed u).property)
  subst v
  have inv := KernelBackward.initial_holds M c s seed p u hh hk hv
  have safe := backward_range M c s p bounds backed u 0 _ inv
  obtain ⟨w,executed,held⟩ := KernelReverseLoop.executes M c s p u 0 _ inv
  have loopRun : step (engine M) (loop .backward c s seed)
      (KernelPrefix.backward M c s seed u).val=some w := by
    simpa [KernelProgram.loop,KernelOps.backward_body,step,FP.Structural.evalOp] using executed
  simp only [List.singleton_append]
  apply traceSafeList_cons_of_step
  · simpa [KernelProgram.loop,stepSafe_unfold,safeOp_unfold,evalOp_unfold,
      KernelOps.backward_body] using safe
  · exact loopRun
  · have incomingOff : KernelReversePointers.off c p (c.N-1) (c.N-1)=
        (fun e j => address c p 0 e.val j.val) := by
      funext e j
      simpa using KernelReversePointers.live c p (c.N-1) (c.N-1) (Nat.le_refl _) e j
    have gateOff : KernelReversePointers.gate c p (c.N-1)=gateAddress c p 0 := by
      simpa using KernelReversePointers.gate_live c p (c.N-1) (Nat.le_refl _)
    apply backward_post M c s seed p bounds backed _ w nonempty held.fixed
    · simpa only [incomingOff] using held.incomingPtr
    · simpa only [gateOff] using held.gatePtr
    · exact held.acc

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelSafety
