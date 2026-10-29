import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONativeAdapter
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open IONativeGraph IONativeAdapter
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prefix_safe (M : Algebra Word) (D : FPCompiled.DotAcc Word) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State Word) (k : Nat)
    (h : FPFused.Invariant M D p initial k current) (hk : k<steps p)
    (access : IOBounds.Accesses p initial bounds) :
    TraceSafe (FPCompiled.backend M D) bounds (prefixKernel p)
      (current.setReg "k" .nat [] (fun _=>k)) := by
  have ha := access.a k hk
  have hb := access.q k hk
  have hs := access.s k hk
  have hz := access.z k hk
  by_cases hno : p.noGroups=Bool.true
  all_goals
    simp only [scaleOffset,zeroOffset,groupIndex,hno,if_true,if_false,Bool.false_eq_true] at hs hz
    simp [TraceSafe,prefixKernel,FPFused.loopPrefix,FPCompiled.body,FPCompiled.source,FPCompiled.typed,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
      bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hno,FPCompiled.backend,
      h.aptrs,h.bptrs,h.acc,h.memory,h.amask,h.sptrs,h.zptrs,h.shift,h.zshift,
      h.bits,h.inf,h.scales,h.zeros,ha,hb,hs,hz]
  all_goals
    first
    | exact fun a b u=>ha (a,b,u)
    | exact ⟨fun a b u=>ha (a,b,u),fun j=>hs (j,()),fun j=>hz (j,())⟩

theorem suffix_safe (M : Algebra Word) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Word) : TraceSafe M bounds (suffixKernel p) s := by
  simp [TraceSafe,suffixKernel,FPFused.suffix,FPCompiled.body,FPCompiled.source,FPCompiled.typed,
    ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem epilogue_safe (M : Algebra Word) (D : FPCompiled.DotAcc Word) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State Word)
    (h : FPFused.Invariant M D p initial (steps p) current)
    (access : IOBounds.Accesses p initial bounds) :
    TraceSafe (FPCompiled.backend M D) bounds (epilogueKernel p) current := by
  simp [TraceSafe,epilogueKernel,FPCompiled.epilogue,FPCompiled.source,FPCompiled.typed,
    ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,h.pidm,h.pidn,h.acc,h.memory,FPCompiled.backend]
  intro r n hr hn
  exact access.c (r,n,()) ⟨hr,hn⟩

theorem block_safe (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State Word) (k : Nat)
    (h : FPFused.Invariant scalar (ISA.dot (mma op)) p initial k current) (hk : k<steps p)
    (access : IOBounds.Accesses p initial bounds) :
    ScheduleSafe bounds (blockJobs actual op p k) current := by
  refine ⟨?_,?_⟩
  · change TraceSafe (model actual op) bounds (prefixKernel p)
      ((held (prefixKernel p) (model actual op) [counter k]).initial current)
    rw [counter_initial,model_eq actual mma ports]
    exact prefix_safe _ _ p bounds initial current k h hk access
  · intro middle _
    refine ⟨?_,?_⟩
    · simpa only [Invocation.Safe,held_initial] using fma_safe actual p bounds middle
    · intro after _
      refine ⟨?_,fun _ _=>trivial⟩
      simpa only [Invocation.Safe,held_initial] using suffix_safe (model actual op) p bounds after

theorem loop_safe (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State Word) (k count : Nat)
    (h : FPFused.Invariant scalar (ISA.dot (mma op)) p initial k current)
    (limit : k+count≤steps p) (access : IOBounds.Accesses p initial bounds) :
    ScheduleSafe bounds (loopJobs actual op p k count) current := by
  induction count generalizing k current with
  | zero=>trivial
  | succ count ih=>
    rw [loopJobs,scheduleSafe_append]
    refine ⟨block_safe actual mma ports op p bounds initial current k h (by omega) access,?_⟩
    intro after ran
    have eq : after=(FPFused.bodyExecution scalar (ISA.dot (mma op)) p initial current k h).val :=
      Option.some.inj (ran.symm.trans (block_run actual mma ports op p initial current k h))
    subst after
    exact ih _ _ (FPFused.bodyInvariant scalar (ISA.dot (mma op)) p initial current k h) (by omega)

theorem program_safe (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (bounds : RegionName → Nat) (pid : Nat) (s : State Word) (positive : 0<steps p)
    (access : IOBounds.Accesses p (GridExecution.atProgram s pid) bounds) :
    ScheduleSafe bounds (programJobs actual op p pid) s := by
  let initial := GridExecution.atProgram s pid
  let start := FPFused.preludeExecution scalar (ISA.dot (mma op)) p initial
  have invariant := FPFused.preludeInvariant scalar (ISA.dot (mma op)) p initial
  have hstart : run (FPCompiled.backend scalar (ISA.dot (mma op))) (FPCompiled.prelude p)
      (GridExecution.atProgram s pid)=some start.val := start.property
  unfold programJobs
  rw [scheduleSafe_append,scheduleSafe_append]
  refine ⟨⟨?_,?_⟩,?_⟩
  · rw [scheduleSafe_singleton]
    change TraceSafe (model actual op) bounds (preludeKernel p) _
    rw [IOExecution.initial_eq]
    exact IOSafety.prelude_safe (model actual op) p bounds initial positive access
  · intro middle ran
    have eq : middle=start.val := by
      have he : runSchedule [IOExecution.launch (preludeKernel p) (model actual op) pid] s=some start.val := by
        simpa only [runSchedule_singleton,Invocation.run,IOExecution.initial_eq,
          FP.Structural.exec,preludeKernel,ComputeKernel.surfaceBody,model_eq actual mma ports] using start.property
      exact Option.some.inj (ran.symm.trans he)
    subst middle
    exact loop_safe actual mma ports op p bounds initial start.val 0 (steps p) invariant (by omega) access
  · intro after ran
    obtain ⟨final,loop,hfinal⟩ := FPFused.loopExecution scalar (ISA.dot (mma op)) p initial start.val
      0 (steps p) invariant
    have he : runSchedule ([IOExecution.launch (preludeKernel p) (model actual op) pid]++
        loopJobs actual op p 0 (steps p)) s=some final := by
      rw [runSchedule_append,runSchedule_singleton]
      simp only [Invocation.run,IOExecution.initial_eq]
      simp only [IOExecution.launch,FP.Structural.exec,preludeKernel,
        ComputeKernel.surfaceBody,model_eq actual mma ports,hstart,Option.bind_some]
      exact (loop_run actual mma ports op p initial start.val 0 (steps p) invariant).trans loop
    have eq : after=final := Option.some.inj (ran.symm.trans he)
    subst after
    rw [scheduleSafe_singleton]
    change TraceSafe (model actual op) bounds (epilogueKernel p) _
    rw [held_initial,model_eq actual mma ports]
    exact epilogue_safe scalar (ISA.dot (mma op)) p bounds initial final (by simpa using hfinal) access

theorem safe (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (x : ChannelValues Word (IOData.inputs c)) (bounds : RegionName → Nat)
    (hb : (IOData.program Word c (jobs actual c i)).InBounds bounds x) (s : State Word) :
    ScheduleSafe bounds (jobs actual c i) s := by
  have w := IOBounds.windows c (jobs actual c i) x bounds hb
  have lists (pids : List Nat) (listed : ∀ pid∈pids,pid<KernelSource.count c.call i)
      (s : State Word) : ScheduleSafe bounds
        (pids.flatMap (programJobs actual (tunings i).opcode (params c.call i))) s := by
    induction pids generalizing s with
    | nil=>trivial
    | cons pid rest ih=>
      rw [List.flatMap_cons,scheduleSafe_append]
      refine ⟨program_safe actual mma ports _ _ bounds pid s (IOBounds.steps_positive c valid i)
        (IOBounds.accesses c valid i bounds w s pid (listed pid (by simp))),?_⟩
      intro after _
      exact ih (fun j hj=>listed j (by simp [hj])) after
  exact lists _ (fun _ h=>List.mem_range.mp h) s

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeSafety
