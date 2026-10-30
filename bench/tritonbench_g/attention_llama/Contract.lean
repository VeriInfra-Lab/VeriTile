import bench.tritonbench_g.attention_llama.Proofs.NativeBinding
import bench.tritonbench_g.attention_llama.Kernels

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama.Checked
abbrev RawScalar := VeriTile.Bench.Repairs.AttentionLlama.Checked.RawScalar

/-- The sole added source condition, evaluated after the empty-output return.
Floating arguments retain the entire old ordinary finite scalar domain. -/
def scalarABICheck (launch : Bool) (raw : RawScalar) : Bool :=
  if launch then match raw with
    | .integer n => decide (-(2^63 : Int)≤n ∧ n<2^64)
    | .float64 _ => Bool.true
  else Bool.true

 theorem empty_scalar_admitted (raw : RawScalar) : scalarABICheck Bool.false raw=Bool.true := rfl
 theorem float_scalar_unchanged (launch : Bool) (bits : BitVec 64) :
    scalarABICheck launch (.float64 bits)=Bool.true := by cases launch <;> rfl
 theorem launched_integer_iff (n : Int) :
    scalarABICheck Bool.true (.integer n)=Bool.true ↔ -(2^63 : Int)≤n ∧ n<2^64 := by
  simp only [scalarABICheck,↓reduceIte,decide_eq_true_eq]

/-- Original input, address and numerical conditions are reused without change.
The launch flag is computed from the actual output shape, not a caller premise. -/
def accepted (B : Nat) (c : VeriTile.Bench.Repairs.AttentionLlama.LoopInvariant.Context)
    (fp8 : Bool) (q k v : VeriTile.Bench.Repairs.AttentionLlama.Checked.TensorMetadata) (a : VeriTile.Bench.Repairs.AttentionLlama.Checked.ScalarMetadata) (raw : RawScalar) : Bool :=
  VeriTile.Bench.Repairs.AttentionLlama.Checked.inputChecks B c fp8 q k v a && scalarABICheck (decide (B*c.NH*c.H*c.D≠0)) raw

 theorem accepted_original (B : Nat) (c : VeriTile.Bench.Repairs.AttentionLlama.LoopInvariant.Context)
    (fp8 : Bool) (q k v : VeriTile.Bench.Repairs.AttentionLlama.Checked.TensorMetadata) (a : VeriTile.Bench.Repairs.AttentionLlama.Checked.ScalarMetadata) (raw : RawScalar)
    (h : accepted B c fp8 q k v a raw=Bool.true) : VeriTile.Bench.Repairs.AttentionLlama.Checked.inputChecks B c fp8 q k v a=Bool.true := by
  simp only [accepted,Bool.and_eq_true] at h
  exact h.1

 theorem accepted_launch_integer (B : Nat) (c : VeriTile.Bench.Repairs.AttentionLlama.LoopInvariant.Context)
    (fp8 : Bool) (q k v : VeriTile.Bench.Repairs.AttentionLlama.Checked.TensorMetadata) (a : VeriTile.Bench.Repairs.AttentionLlama.Checked.ScalarMetadata) (n : Int)
    (nonempty : B*c.NH*c.H*c.D≠0)
    (h : accepted B c fp8 q k v a (.integer n)=Bool.true) : -(2^63 : Int)≤n ∧ n<2^64 := by
  simp only [accepted,Bool.and_eq_true] at h
  have hs := h.2
  simpa only [scalarABICheck,decide_eq_true nonempty,↓reduceIte,decide_eq_true_eq] using hs

#axiomsClean empty_scalar_admitted
#axiomsClean float_scalar_unchanged
#axiomsClean launched_integer_iff
#axiomsClean accepted_original
#axiomsClean accepted_launch_integer
#specNonCircular accepted avoiding [FP.Structural.exec,FP.Structural.run]
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
