import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideCompletion
import bench.tritonbench_g.spinning_lock_reduction.Proofs.FP
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Real

/-! Coordinate-preserving lazy-negative view contract. Physical backing storage,
addresses, allocation identity and caller-supplied strides are preserved. The
exact source representation operations are an FP32 sign-bit XOR and int32
negation for the ready word. They are involutions, including both zeros and all
NaN payloads. No Torch implementation proof or floating approximation is used.
-/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton

structure ViewFlags where
  partialNeg : Bool := false
  output : Bool := false
  locks : Bool := false
  deriving DecidableEq

structure SignEncoding (V : Type) where
  flip : Bool → V → V
  twice : ∀ flag x, flip flag (flip flag x) = x

def fpSign (flag : Bool) (x : FP.Value .fp32) : FP.Value .fp32 :=
  if flag then ⟨x.bits ^^^ (BitVec.ofNat FP.Format.fp32.width 0x80000000)⟩ else x

theorem fpSign_twice (flag : Bool) (x : FP.Value .fp32) : fpSign flag (fpSign flag x) = x := by
  cases flag <;> cases x <;> simp [fpSign, BitVec.xor_assoc]

def fpEncoding : SignEncoding (FP.Value .fp32) := ⟨fpSign, fpSign_twice⟩

def realEncoding : SignEncoding ℝ where
  flip flag x := if flag then -x else x
  twice := by intro flag x; cases flag <;> simp

def lockWord (flag : Bool) (x : I32) : I32 := if flag then -x else x

@[simp] theorem lockWord_twice (flag : Bool) (x : I32) : lockWord flag (lockWord flag x) = x := by
  cases flag <;> simp [lockWord]

theorem lock_ready_encoding (flag : Bool) (raw : I32) :
    lockWord flag raw = 1 ↔ raw = lockWord flag 1 := by
  constructor
  · intro hh
    have := congrArg (lockWord flag) hh
    simpa using this
  · intro hh; rw [hh, lockWord_twice]

theorem fp_physical_zero_is_negative_zero :
    fpSign true fp32.zero = FP.Value.ofBits .fp32 0x80000000 := by decide +kernel

theorem fp_logical_zero_readback (flag : Bool) :
    fpSign flag (fpSign flag fp32.zero) = fp32.zero := fpSign_twice flag _

def interpretMemory (encoding : SignEncoding V) (flags : ViewFlags) (mem : Memory V) : Memory V :=
  ⟨fun a => encoding.flip flags.partialNeg (mem.partials a),
    fun a => encoding.flip flags.output (mem.output a),
    fun a => lockWord flags.locks (mem.locks a)⟩

@[simp] theorem interpretMemory_twice (encoding : SignEncoding V) (flags : ViewFlags) (mem : Memory V) :
    interpretMemory encoding flags (interpretMemory encoding flags mem) = mem := by
  cases mem
  simp only [interpretMemory, encoding.twice, lockWord_twice]

def interpretValues (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) :
    VeriTile.Bench.TritonBenchG.SpinningLock.State V :=
  { s with mem := interpretMemory encoding flags s.mem }

@[simp] theorem interpretValues_twice (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) :
    interpretValues encoding flags (interpretValues encoding flags s) = s := by
  cases s
  simp [interpretValues]

def interpretState (encoding : SignEncoding V) (flags : ViewFlags) (s : State V) : State V :=
  { s with values := interpretValues encoding flags s.values }

@[simp] theorem interpretState_twice (encoding : SignEncoding V) (flags : ViewFlags) (s : State V) :
    interpretState encoding flags (interpretState encoding flags s) = s := by
  cases s
  simp [interpretState]

/-- Conjugated value executor. The following source-operation equations expose
its individual physical reads, stores and ready word; this is not an assumed
whole-output relation. Accumulators are logical FP32 throughout. -/
def executePhysical (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (e : Event) (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) :=
  interpretValues encoding flags (execute b e (interpretValues encoding flags s))

theorem physical_add (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) (p l : Nat) (a : Int) :
    (executePhysical b encoding flags (.add p l a) s).acc p l =
      b.add (s.acc p l) (encoding.flip flags.partialNeg (s.mem.partials a)) := by
  simp [executePhysical, interpretValues, interpretMemory, execute]

theorem physical_publish (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) (p l : Nat) (a : Int) :
    (executePhysical b encoding flags (.publish p l a) s).mem.partials a =
      encoding.flip flags.partialNeg (s.acc p l) := by
  simp [executePhysical, interpretValues, interpretMemory, execute]

theorem physical_output (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) (p l : Nat) (a : Int) :
    (executePhysical b encoding flags (.output p l a) s).mem.output a =
      encoding.flip flags.output (s.acc p l) := by
  simp [executePhysical, interpretValues, interpretMemory, execute]

theorem physical_release (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) (a : Int) :
    (executePhysical b encoding flags (.release a) s).mem.locks a = lockWord flags.locks 1 := by
  simp [executePhysical, interpretValues, interpretMemory, execute]

theorem physical_poll (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (s : VeriTile.Bench.TritonBenchG.SpinningLock.State V) (a : Int) :
    executePhysical b encoding flags (.poll a) s = s := by
  simp [executePhysical, execute, cas11_returns_old_and_preserves_bits]

/-- Physical decode uses actual raw lock bits, interpreted through the explicit
lock_ready_encoding equivalence. All addresses and native integer operations
are exactly nativeDecode. -/
def physicalStep (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (p : Nat) (s : State V) : Event × State V :=
  let (event, ctl) := nativeDecode h p (interpretState encoding flags s)
  (event, ⟨executePhysical b encoding flags event s.values, put s.pc p ctl⟩)

def physicalRun (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (schedule : List Nat) (s : State V) : List Event × State V :=
  let result := nativeRun b h schedule (interpretState encoding flags s)
  (result.1, interpretState encoding flags result.2)

theorem physicalRun_nil (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (s : State V) : physicalRun b encoding flags h [] s = ([], s) := by
  simp [physicalRun, nativeRun]

/-- Thus the conjugated run is precisely the list recursion over physical source
steps, not merely a desired final-memory equation. -/
theorem physicalRun_cons (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (p : Nat) (ps : List Nat) (s : State V) :
    physicalRun b encoding flags h (p :: ps) s =
      let first := physicalStep b encoding flags h p s
      let rest := physicalRun b encoding flags h ps first.2
      (first.1 :: rest.1, rest.2) := by
  simp [physicalRun, physicalStep, nativeRun, nativeStep, executePhysical, interpretState,
    interpretValues]

def interpretResult (encoding : SignEncoding V) (flags : ViewFlags) (result : WrapperResult V) : WrapperResult V :=
  { result with final := interpretState encoding flags result.final }

@[simp] theorem interpretResult_twice (encoding : SignEncoding V) (flags : ViewFlags) (result : WrapperResult V) :
    interpretResult encoding flags (interpretResult encoding flags result) = result := by
  cases result
  simp [interpretResult]

def physicalLaunch (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (schedule : List Nat) (raw : Memory V) : WrapperResult V :=
  interpretResult encoding flags (launch b h schedule (interpretMemory encoding flags raw))

theorem physical_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (encoding : SignEncoding V) (flags : ViewFlags) (h : HostArgs) (buf : Buffers h)
    (raw : Memory V) (fresh : ∀ a, (interpretMemory encoding flags raw).locks a = 0) :
    ∃ schedule result, result = physicalLaunch b encoding flags h schedule raw ∧
      CompleteFacts b h buf schedule (interpretMemory encoding flags raw) (interpretResult encoding flags result) := by
  obtain ⟨schedule, result, facts⟩ := fresh_complete_backend b hadd h buf _ fresh
  refine ⟨schedule, interpretResult encoding flags result, ?_, ?_⟩
  · rw [facts.execution]; rfl
  · simpa using facts

#axiomsClean fpSign_twice
#axiomsClean physicalRun_cons
#axiomsClean physical_complete_backend
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide
