import bench.tritonbench_g.spinning_lock_reduction.Proofs.Support.SourceAdapter
import VeriTile.Triton.DSL.PastedControlSource

/-! Complete source programs use the shared control-flow frontend. The
checked adapter retains the original typed SC execution model and rejects
unsupported source forms. Compilation witnesses cover the full programs;
spinning, barriers, atomic results and all nine rounds remain explicit. -/
namespace VeriTile.Bench.Repairs.SpinningLock
open SourceReader VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton
set_option maxRecDepth 50000
set_option maxHeartbeats 8000000

/-- Runtime pointer/scalar types are supplied by the repaired wrapper. Its
input gate rejects negative/conjugate tensor views before querying flags. -/
def callSignature : CallSignature where
  pointers := [("P",.fp32),("C",.fp32),("locks",.i32)]
  scalars := [("num_sms",.integer),("k",.integer),("M",.integer),("N",.integer),
    ("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","PHASE","P_NEG","C_NEG","LOCK_NEG"]
  readPorts := ["P","C","locks"]
  writePorts := ["P","C","locks"]

/-- One source, specialized by PHASE at execution. The accumulator persists
across all nine rounds. Producer and consumer kernels differ only by PHASE. -/
def reductionSource : VeriTile.Triton.DSL.PastedControlSource.Invocation :=
    with_control_call callSignature constexpr_bools [P_NEG := false,C_NEG := false,LOCK_NEG := false] in triton {
    pid = tl.program_id(0).to(tl.int64)
    tl.static_assert(PHASE == 1 or PHASE == 2)
    is_reducer = pid.to(tl.uint64) % tl.cast(k, tl.uint64) == 0
    if (PHASE == 1 and is_reducer) or (PHASE == 2 and not is_reducer):
        return
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization

    lock_ready = -1 if LOCK_NEG else 1

    # Perform reduction for every kth pid
    for iters in range(1, 10):
        if is_reducer:
            next_pid = pid + 1

            while next_pid.to(tl.uint64) < pid.to(tl.uint64) + tl.cast(k, tl.uint64) and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, lock_ready, lock_ready) != lock_ready:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
                rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                if P_NEG:
                    acc1 = (acc1.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
                acc += acc1

                next_pid += 1

        # Store results using temporary storage P for every k-1 pids
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if P_NEG else acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, lock_ready)

        # Store final results in C
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if C_NEG else acc, mask=mask)
}

/-- Fold constexpr tile area before the live 64-bit program-index multiply.
The complete CAS/barrier/control source is retained. -/
def areaSource : VeriTile.Triton.DSL.PastedControlSource.Invocation :=
    with_control_call callSignature constexpr_bools [P_NEG := false,C_NEG := false,LOCK_NEG := false] in triton {
    pid = tl.program_id(0).to(tl.int64)
    tl.static_assert(PHASE == 1 or PHASE == 2)
    is_reducer = pid.to(tl.uint64) % tl.cast(k, tl.uint64) == 0
    if (PHASE == 1 and is_reducer) or (PHASE == 2 and not is_reducer):
        return
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization

    lock_ready = -1 if LOCK_NEG else 1

    # Perform reduction for every kth pid
    for iters in range(1, 10):
        if is_reducer:
            next_pid = pid + 1

            while next_pid.to(tl.uint64) < pid.to(tl.uint64) + tl.cast(k, tl.uint64) and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, lock_ready, lock_ready) != lock_ready:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
                rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                P_ = P + next_pid * (BLOCK_SIZE_M * BLOCK_SIZE_N) + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                if P_NEG:
                    acc1 = (acc1.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
                acc += acc1

                next_pid += 1

        # Store results using temporary storage P for every k-1 pids
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            P_ = P + pid * (BLOCK_SIZE_M * BLOCK_SIZE_N) + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if P_NEG else acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, lock_ready)

        # Store final results in C
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if C_NEG else acc, mask=mask)
}

/-- Full checked lowering. The value is inferred from the successful decode;
there is no fallback program for an unsupported statement. -/
abbrev kernelDecoded : { body : List Statement // SourceAdapter.decode reductionSource = .ok body } := by
  refine ⟨?_, ?_⟩
  case refine_2 => rfl
abbrev areaKernelDecoded : { body : List Statement // SourceAdapter.decode areaSource = .ok body } := by
  refine ⟨?_, ?_⟩
  case refine_2 => rfl

def kernel : List Statement := kernelDecoded.val
def areaKernel : List Statement := areaKernelDecoded.val

/-- Public source execution. The option records source decoding/static-assert
failure; compilation success is proved, never assumed as an input contract. -/
def runKernel (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (schedule : List Nat) (mem : Memory V) :=
  compileAndRun b kernel h stage schedule mem

end VeriTile.Bench.Repairs.SpinningLock
