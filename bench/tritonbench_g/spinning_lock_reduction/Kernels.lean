import bench.tritonbench_g.spinning_lock_reduction.SourceAdapter

/-! Complete source programs use the shared control-flow frontend. The
checked adapter retains the original typed SC execution model and rejects
unsupported source forms. Compilation witnesses cover the full programs;
spinning, barriers, atomic results and all nine rounds remain explicit. -/
namespace VeriTile.Bench.Repairs.SpinningLock
open SourceReader VeriTile.Bench.TritonBenchG.SpinningLock
set_option maxRecDepth 50000
set_option maxHeartbeats 8000000

/-- One source, specialized by PHASE at execution. The accumulator persists
across all nine rounds. Producer and consumer kernels differ only by PHASE. -/
def source : VeriTile.Triton.ControlSource.Program := source_program triton {
  pid = tl.cast(tl.program_id(0), tl.int64)
  tl.static_assert(PHASE == 1 or PHASE == 2)
  is_reducer = tl.cast(pid, tl.uint64) % tl.cast(k, tl.uint64) == 0
  if (PHASE == 1 and is_reducer) or (PHASE == 2 and not is_reducer) {
    return
  }
  pid_m = pid // num_sms
  pid_n = pid % num_sms
  acc = tl.zeros([BM, BN], dtype=tl.float32)
  lock_ready = 1
  for iters in range(1, 10) {
    if is_reducer {
      next_pid = pid + 1
      while tl.cast(next_pid, tl.uint64) < tl.cast(pid, tl.uint64) + tl.cast(k, tl.uint64) and next_pid < num_sms {
        while tl.atomic_cas(locks + next_pid, lock_ready, lock_ready) != lock_ready {
          pass
        }
        rm1 = tl.cast(tl.arange(0, BM), tl.int64)
        rn1 = tl.cast(tl.arange(0, BN), tl.int64)
        P_ = P + next_pid * BM * BN + rm1[:, None] * BN + rn1[None, :]
        acc1 = tl.load(P_)
        acc += acc1
        next_pid += 1
      }
    } else {
      rm1 = tl.cast(tl.arange(0, BM), tl.int64)
      rn1 = tl.cast(tl.arange(0, BN), tl.int64)
      P_ = P + pid * BM * BN + rm1[:, None] * BN + rn1[None, :]
      tl.store(P_, acc)
      tl.debug_barrier()
      tl.atomic_xchg(locks + pid, lock_ready)
    }
    rm = pid_m * BM + tl.cast(tl.arange(0, BM), tl.int64)
    rn = pid_n * BN + tl.cast(tl.arange(0, BN), tl.int64)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C_, acc, mask=mask)
  }
}

/-- Fold constexpr tile area before the live 64-bit program-index multiply.
The complete CAS/barrier/control source is retained. -/
def areaSource : VeriTile.Triton.ControlSource.Program := source_program triton {
  pid = tl.cast(tl.program_id(0), tl.int64)
  tl.static_assert(PHASE == 1 or PHASE == 2)
  is_reducer = tl.cast(pid, tl.uint64) % tl.cast(k, tl.uint64) == 0
  if (PHASE == 1 and is_reducer) or (PHASE == 2 and not is_reducer) {
    return
  }
  pid_m = pid // num_sms
  pid_n = pid % num_sms
  acc = tl.zeros([BM, BN], dtype=tl.float32)
  lock_ready = 1
  for iters in range(1, 10) {
    if is_reducer {
      next_pid = pid + 1
      while tl.cast(next_pid, tl.uint64) < tl.cast(pid, tl.uint64) + tl.cast(k, tl.uint64) and next_pid < num_sms {
        while tl.atomic_cas(locks + next_pid, lock_ready, lock_ready) != lock_ready {
          pass
        }
        rm1 = tl.cast(tl.arange(0, BM), tl.int64)
        rn1 = tl.cast(tl.arange(0, BN), tl.int64)
        P_ = P + next_pid * (BM * BN) + rm1[:, None] * BN + rn1[None, :]
        acc1 = tl.load(P_)
        acc += acc1
        next_pid += 1
      }
    } else {
      rm1 = tl.cast(tl.arange(0, BM), tl.int64)
      rn1 = tl.cast(tl.arange(0, BN), tl.int64)
      P_ = P + pid * (BM * BN) + rm1[:, None] * BN + rn1[None, :]
      tl.store(P_, acc)
      tl.debug_barrier()
      tl.atomic_xchg(locks + pid, lock_ready)
    }
    rm = pid_m * BM + tl.cast(tl.arange(0, BM), tl.int64)
    rn = pid_n * BN + tl.cast(tl.arange(0, BN), tl.int64)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C_, acc, mask=mask)
  }
}

/-- Full checked lowering. The value is inferred from the successful decode;
there is no fallback program for an unsupported statement. -/
abbrev kernelDecoded : { body : List Statement // SourceAdapter.decode source = .ok body } := by
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
