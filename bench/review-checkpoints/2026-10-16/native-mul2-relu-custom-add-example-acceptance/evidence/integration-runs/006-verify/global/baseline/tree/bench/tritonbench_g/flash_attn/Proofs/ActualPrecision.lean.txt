import bench.tritonbench_g.flash_attn.Kernels
import bench.tritonbench_g.flash_attn.Proofs.Formula

/-! Independent typed AST prescriptions for the unmodified source. The fp64
query multiplication retains its wider precision before the explicit fp16
cast. The two stores have exactly the casts written by the source. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.ActualPrecision
open VeriTile.Triton TypedSource Formula
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def body (input : Storage) (v : Bool) (Q K V L O : RegionName) (p : Layout) :=
  (actualKernel input v Q K V L O p.sm_scale
    0 p.stride_q_head p.stride_q_seqlen p.stride_q_dim
    0 0 p.stride_k_seqlen p.stride_k_dim 0 0 0 p.stride_v_dim
    0 0 p.stride_o_seqlen p.stride_o_dim 0 0 p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).surfaceBody

def queryLoad (input : Storage) (M D : Nat) : ComputeStmt :=
  let mem (d : TileDType) : MemAccess d [M,D] :=
    .blockPtr (.ref .blockPtr [M,D] "Q_block_ptr") []
  match input with
  | .fp32 => .assign .real [M,D] "q" (.compute (.load .fp32 (mem .real) .none .unspecified))
  | .fp64 => .assign .real [M,D] "q" (.compute (.alg .fp64 (.load .real (mem .real) .none .unspecified)))
  | .fp16 => .assign .fp16 [M,D] "q" (.alg (.load .fp16 (mem .fp16) .none .unspecified))
  | .bf16 => .assign .bf16 [M,D] "q" (.alg (.load .bf16 (mem .bf16) .none .unspecified))
  | .f8e4 => .assign .f8e4 [M,D] "q" (.alg (.load .f8e4 (mem .f8e4) .none .unspecified))
  | .f8e5 => .assign .f8e5 [M,D] "q" (.alg (.load .f8e5 (mem .f8e5) .none .unspecified))

def queryScale (input : Storage) (M D : Nat) : ComputeStmt :=
  let scale : Op .real [] := .ref .real [] "qk_scale"
  let q : Op .real [M,D] := match input with
    | .fp32 | .fp64 => .ref .real [M,D] "q"
    | .fp16 => .castFloat .fp16 .real (.ref .fp16 [M,D] "q")
    | .bf16 => .castFloat .bf16 .real (.ref .bf16 [M,D] "q")
    | .f8e4 => .castFloat .f8e4 .real (.ref .f8e4 [M,D] "q")
    | .f8e5 => .castFloat .f8e5 .real (.ref .f8e5 [M,D] "q")
  let value : Op .real [M,D] := match input with
    | .fp64 => .withPrecision .real .fp64 (.mul .real .scalarR q (.withPrecision .real .fp32 scale))
    | _ => .withPrecision .real .fp32 (.mul .real .scalarR q scale)
  .assign .fp16 [M,D] "q" (.alg (.castFloat .real .fp16 value))

def lStore (M : Nat) : ComputeStmt :=
  .store .real [M] (.ptr (.ref .ptr [M] "l_ptr"))
    (.compute (.alg .fp32 (.add .real (.consSame .nil) (.ref .real [M] "max")
      (.log2 (.ref .real [M] "denom"))))) .none

def oStore (M D : Nat) : ComputeStmt :=
  .store .fp16 [M,D] (.blockPtr (.ref .blockPtr [M,D] "O_block_ptr") [])
    (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [M,D] "out_buffer")))) .none

/-- All six source-admitted floating Q labels keep the exact scale, load,
pre-cast multiplication and both source stores, including both causal modes. -/
theorem source_profiles (input : Storage) (v : Bool) (Q K V L O : RegionName) (p : Layout) :
    (body input v Q K V L O p)[11]? = some
      (.assign .real [] "qk_scale" (.alg (.mul .real .nil (.const p.sm_scale) (.const 1.44269504)))) ∧
    (body input v Q K V L O p)[12]? = some (queryLoad input p.BLOCK_M p.DIM) ∧
    (body input v Q K V L O p)[13]? = some (queryScale input p.BLOCK_M p.DIM) ∧
    (body input v Q K V L O p)[19]? = some (lStore p.BLOCK_M) ∧
    (body input v Q K V L O p)[21]? = some (oStore p.BLOCK_M p.DIM) := by
  cases input <;> cases v <;> exact ⟨rfl,rfl,rfl,rfl,rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.ActualPrecision
