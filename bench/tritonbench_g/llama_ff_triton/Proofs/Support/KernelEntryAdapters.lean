import bench.tritonbench_g.llama_ff_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Source
open VeriTile.Triton
noncomputable section

def half (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel :=
  halfKernel A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS

def byte (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel :=
  byteKernel A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS

def halfCandidate (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel :=
  halfCandidateKernel A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS

def byteCandidate (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel :=
  byteCandidateKernel A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS

def kernel (fp8 : Bool) (M N K : Nat) (EPS : ℝ) : ComputeKernel :=
  (if fp8 then byte else half) "A" "W1" "W3" "OUT" "RMS"
    M N K K 1 1 K 1 K N 1 1 16 16 64 EPS

/-- Repaired existing candidate on the exact selected launch profile. -/
def candidate (fp8 : Bool) (M N K : Nat) (EPS : ℝ) : ComputeKernel :=
  (if fp8 then byteCandidate else halfCandidate) "A" "W1" "W3" "OUT" "RMS"
    M N K K 1 1 K 1 K N 1 1 16 16 64 EPS

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Source
