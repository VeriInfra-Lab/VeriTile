import bench.tritonbench_g.triton_matmul.Sources.Parameters

/-! Independent host configuration and Real matrix specification for the repaired source. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.SourceScope
open VeriTile.Triton FPValue
noncomputable section

def blockK : Storage → Nat | .half => 64 | .e4m3 => 128
def stages : Storage → Nat | .half => 3 | .e4m3 => 4
def warps (_ : Storage) : Nat := 8

def parameters (A B C : RegionName) (st : Storage)
    (M N K sam sak sbk sbn : Nat) : Parameters where
  A := A; B := B; C := C
  M := M; N := N; K := K
  stride_am := sam; stride_ak := sak; stride_bk := sbk; stride_bn := sbn
  stride_cm := max N 1; stride_cn := 1
  BLOCK_SIZE_M := 128; BLOCK_SIZE_N := 256; BLOCK_SIZE_K := blockK st
  GROUP_SIZE_M := 8
  numKBlocks := (K+blockK st-1)/blockK st

def grid (M N : Nat) := ((M+127)/128)*((N+255)/256)

/-- Independent mathematical target, not source-derived accumulator values. -/
def mathematical (M N K : Nat) (A : Fin M → Fin K → ℝ)
    (B : Fin K → Fin N → ℝ) (i : Fin M) (j : Fin N) : ℝ :=
  ∑ k : Fin K, A i k*B k j

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.SourceScope
