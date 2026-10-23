import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPComplete

/-! Actual source configuration, selected default precision, and independent
logical target. These definitions do not identify Torch logical view values
with raw storage. The original host fails that identification for negative
views, as recorded by LogicalViewDefect. -/
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.SourceScope
open VeriTile.Triton FP.Structural FPValue
noncomputable section

def stages : Nat := 3
def warps : Nat := 8

def parameters (A B C : RegionName) (M N K sam sak sbk sbn : Nat)
    (enabled : Bool) : Parameters where
  A := A; B := B; C := C
  M := M; N := N; K := K
  stride_am := sam; stride_ak := sak; stride_bk := sbk; stride_bn := sbn
  stride_cm := max N 1; stride_cn := 1
  BLOCK_SIZE_M := 128; BLOCK_SIZE_N := 256; BLOCK_SIZE_K := 64
  GROUP_SIZE_M := 8
  ACTIVATION := enabled

/-- The production wrapper makes no input conversion or dtype restriction.
The selected unspecified-input profile is FP32, with explicit half output. -/
def kernel (A B C : RegionName) (M N K sam sak sbk sbn : Nat)
    (enabled : Bool) : ComputeKernel :=
  source (parameters A B C M N K sam sak sbk sbn enabled) .single

def activationEnabled (activation : String) : Bool := activation == "leaky_relu"
def grid (M N : Nat) := ((M+127)/128)*((N+255)/256)

/-- Independent matrix product and original left-side leaky-ReLU slope. -/
def mathematical (M N K : Nat) (enabled : Bool) (A : Fin M → Fin K → ℝ)
    (B : Fin K → Fin N → ℝ) (i : Fin M) (j : Fin N) : ℝ :=
  let x := ∑ k : Fin K, A i k*B k j
  if enabled then if 0≤x then x else (1/100 : ℝ)*x else x

theorem config_table (A B C : RegionName) (M N K sam sak sbk sbn : Nat)
    (enabled : Bool) :
    let p := parameters A B C M N K sam sak sbk sbn enabled
    BM p=128 ∧ BN p=256 ∧ BK p=64 ∧ p.GROUP_SIZE_M=8 ∧
      stages=3 ∧ warps=8 ∧ dtype .single=.fp32 ∧ trips p=(K+63)/64 := by
  simp [parameters,BM,BN,BK,stages,warps,dtype,trips]

#axiomsClean config_table
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.SourceScope
