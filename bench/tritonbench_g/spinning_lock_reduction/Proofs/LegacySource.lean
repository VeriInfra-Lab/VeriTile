import bench.tritonbench_g.spinning_lock_reduction.SourceSyntax

/-! Frozen selected quotations for complete program preservation checks.
The canonical shared source lives in Kernels.lean. These fixtures retain the
original false P_NEG/C_NEG/LOCK_NEG specialization and all control effects. -/
namespace VeriTile.Bench.Repairs.SpinningLock.SourceLegacy
open SourceReader SourceSyntax VeriTile.Bench.TritonBenchG.SpinningLock

/-- One source, specialized by PHASE at execution. The accumulator persists
across all nine rounds. Producer and consumer kernels differ only by PHASE. -/
def kernel : List Statement := spinning_triton {
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
def areaKernel : List Statement := spinning_triton {
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

/-- Public source execution. The option records source decoding/static-assert
failure; compilation success is proved, never assumed as an input contract. -/
def runKernel (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (schedule : List Nat) (mem : Memory V) :=
  compileAndRun b kernel h stage schedule mem

end VeriTile.Bench.Repairs.SpinningLock.SourceLegacy
