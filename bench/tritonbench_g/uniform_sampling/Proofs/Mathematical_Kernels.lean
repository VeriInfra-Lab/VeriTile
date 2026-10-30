/-
Source-audit target: uniform_kernel
Source-audit Lean: uniformSource%
Source-audit args: storage=tl.float32, version=original
Source-audit params: to=upper
Source-audit materializations: {"kind":"direct_uniform_identity_v1","source_sha256":"d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1","lean":{"source":"uniformSource%","args":{"storage":"tl.float32","version":"original"}},"regions":{"out_ptr":"tl.float32"},"bindings":[{"python":"philox_seed","local":"philox_seed_arg","kind":"runtime_integer","dtype":"tl.int64","initial_cast":"philox_seed"},{"python":"philox_offset","local":"philox_offset_arg","kind":"runtime_integer","dtype":"tl.int64","initial_cast":"philox_offset"},{"python":"BLOCK","local":"BLOCK_arg","kind":"constexpr_identity","adapter":"_root_.VeriTile.Triton.Random.i32ArangeSize","adapter_sha256":"61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04","proof":{"function":"arange_domain","binder":"hBlock"}}],"helper":{"python":"uint_to_uniform_float","builtin":"tl.uint_to_uniform_float","binding":"original_try_attributeerror"},"params":{"to":"upper"}}

The strict source comparison validates all six explicit storage ASTs, with
the 24 production audit gates run for the declared fp32 profile. The macro
retains every original kernel statement, including all
four tuple results, the modular augmented assignment, and all four stores.
The three prefix assignments materialize the two runtime signed64 parameters
and the identity constexpr adapter. The explicit final casts represent native
destination conversion. The helper declaration records the original module's
try/except binding; it is not an arbitrary bare-name substitution.
-/
import VeriTile.Triton.DSL
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_SourceDomain

namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedSource
open Lean VeriTile.Triton

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq

theorem arange_domain {B : Nat} (hB : SourceDomain.LegalBlock B) :
    Random.ArangeSizeDomain B := by
  have hf := SourceDomain.legalBlock_fields hB
  obtain ⟨k, _, hk⟩ := hB
  exact ⟨hf.1, hf.2.1, k, hk⟩

syntax "uniformSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(uniformSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let index : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| i4 = BLOCK_arg * tl.program_id(0) + tl.arange(0, BLOCK_arg))
      else
        `(tritonStmt| i4 = tl.program_id(0) * BLOCK_arg + tl.arange(0, BLOCK_arg))
    `(fun (out_ptr : RegionName) (N : Nat) (philox_seed philox_offset : Int)
      (from_ upper : ℝ) (BLOCK : Nat) (hBlock : SourceDomain.LegalBlock BLOCK) =>
      ComputeKernel.mk [] [out_ptr] (triton {
  tl.region out_ptr = $storage
  tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
  philox_seed_arg = $((philox_seed : Int))
  tl.source_integer philox_seed_arg = tl.int64
  philox_offset_arg = $((philox_offset : Int))
  tl.source_integer philox_offset_arg = tl.int64
  BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK (arange_domain hBlock))
  philox_seed = philox_seed_arg.to(tl.int64)
  philox_offset = philox_offset_arg.to(tl.int64)
  c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
  c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
  $index
  c0 += i4
  _O = c0 * 0
  r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
  r0 = uint_to_uniform_float(r0) * ($((upper : ℝ)) - $((from_ : ℝ))) + $((from_ : ℝ))
  r1 = uint_to_uniform_float(r1) * ($((upper : ℝ)) - $((from_ : ℝ))) + $((from_ : ℝ))
  r2 = uint_to_uniform_float(r2) * ($((upper : ℝ)) - $((from_ : ℝ))) + $((from_ : ℝ))
  r3 = uint_to_uniform_float(r3) * ($((upper : ℝ)) - $((from_ : ℝ))) + $((from_ : ℝ))
  off_0 = tl.program_id(0) * BLOCK_arg * 4 + tl.arange(0, BLOCK_arg)
  off_1 = off_0 + BLOCK_arg
  off_2 = off_1 + BLOCK_arg
  off_3 = off_2 + BLOCK_arg
  tl.store(out_ptr + off_0, (r0).to($storage:tritonDType), mask=off_0 < $(N), eviction_policy="evict_first")
  tl.store(out_ptr + off_1, (r1).to($storage:tritonDType), mask=off_1 < $(N), eviction_policy="evict_first")
  tl.store(out_ptr + off_2, (r2).to($storage:tritonDType), mask=off_2 < $(N), eviction_policy="evict_first")
  tl.store(out_ptr + off_3, (r3).to($storage:tritonDType), mask=off_3 < $(N), eviction_policy="evict_first")
}).surfaceBody)

abbrev sourceType := RegionName → Nat → Int → Int → ℝ → ℝ →
  (B : Nat) → SourceDomain.LegalBlock B → ComputeKernel

def source (storage : Storage) (swapped : Bool) : sourceType :=
  match storage, swapped with
  | .fp32, Bool.false => uniformSource%[tl.float32, original]
  | .fp32, Bool.true => uniformSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => uniformSource%[tl.float16, original]
  | .fp16, Bool.true => uniformSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => uniformSource%[tl.bfloat16, original]
  | .bf16, Bool.true => uniformSource%[tl.bfloat16, indexCommuted]
  | .fp64, Bool.false => uniformSource%[tl.float64, original]
  | .fp64, Bool.true => uniformSource%[tl.float64, indexCommuted]
  | .f8e4, Bool.false => uniformSource%[tl.float8e4nv, original]
  | .f8e4, Bool.true => uniformSource%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => uniformSource%[tl.float8e5, original]
  | .f8e5, Bool.true => uniformSource%[tl.float8e5, indexCommuted]

#axiomsClean arange_domain

end VeriTile.Bench.TritonBenchG.UniformSampling.TypedSource
