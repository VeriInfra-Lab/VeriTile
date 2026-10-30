import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem update_congr {s t : State α} {n : RegName} {d : TileDType} {sh : TileShape}
    {v w : Values α d sh} (same : s=t) (values : v=w) : s.setReg n d sh v=t.setReg n d sh w := by
  cases same
  cases values
  rfl

private theorem int_ite_nat (cond : Prop) [Decidable cond] (a b : Nat) :
    (if cond then (a : Int) else (b : Int))=((if cond then a else b : Nat) : Int) := by
  split <;> rfl

private theorem masked_sum_nat (c d : Prop) [Decidable c] [Decidable d] (a b : Nat) :
    Int.toNat ((if c then (a : Int) else 0)+(if d then (b : Int) else 0))=
      (if c then a else 0)+(if d then b else 0) := by
  split_ifs <;> simp

def floatValues (profile : Storage) {sh : TileShape} (value : TileIndex sh → α) : Values α (dtype profile) sh :=
  match profile with
  | .fp16 => value
  | .f8e4 => value

def bodyResult (P : Source.Arithmetic α α) (p : Parameters) (initial current : State α)
    (tile block : Nat) : State α :=
  let s := current.setReg "ki" .nat [] (fun _ => block)
  let s := s.setReg "raw_k" .int [p.problem.profile.blockK] (fun i => (p.problem.k block i.1.val : Int))
  let s := s.setReg "offs_k" .int [p.problem.profile.blockK] (fun i => (clamp p.problem.K (p.problem.k block i.1.val) : Int))
  let s := s.setReg "a_ptrs" .ptr [128,p.problem.profile.blockK] (fun i => ("A",aOffset p tile block i))
  let s := s.setReg "b_ptrs" .ptr [p.problem.profile.blockK,256] (fun i => ("B",bOffset p tile block i))
  let s := s.setReg "a" (dtype p.problem.profile) [128,p.problem.profile.blockK]
    (floatValues p.problem.profile (aValues P p initial tile block))
  let s := s.setReg "b" (dtype p.problem.profile) [p.problem.profile.blockK,256]
    (floatValues p.problem.profile (bValues P p initial tile block))
  s.setReg "accumulator" .real [128,256] (fun i => accumulation P p initial tile (block+1) i.1 i.2.1)

theorem bodyExecution (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (tile block : Nat) (h : InnerInvariant P p initial tile block current) :
    run (backend P) (inner p) (current.setReg "ki" .nat [] (fun _ => block))=
      some (bodyResult P p initial current tile block) := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile
  case fp16 =>
    let p : Parameters := ⟨⟨M,N,K,SMS,.fp16⟩,am,ak,bk,bn,cm,cn⟩
    change run (backend P) (inner p) (current.setReg "ki" .nat [] (fun _ => block))=
      some (bodyResult P p initial current tile block)
    have h : InnerInvariant P p initial tile block current := h
    simp [p,inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,integralFloorDiv,integralMod,backend,matrix,h.offAM,h.offBN,h.offK,h.acc,
      h.memory,bodyResult,accumulation,aValues,bValues,aOffset,bOffset,readCell,floatValues,dtype,
      Problem.k,clamp,Broadcast.leftIndex,Broadcast.rightIndex,RealMath.signed_tail_mask]
    try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    try simp only [← Int.natCast_mul,← Int.natCast_add,int_ite_nat,Int.ofNat_lt,Int.toNat_natCast]
    repeat' apply update_congr
    all_goals first
    | rfl
    | funext i
      try simp [p,aValues,bValues,aOffset,bOffset,readCell,clamp,Problem.k,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        -Nat.cast_mul,-Nat.cast_add,-Nat.cast_ite,
        ← Int.natCast_mul,← Int.natCast_add,int_ite_nat,masked_sum_nat]
  case f8e4 =>
    let p : Parameters := ⟨⟨M,N,K,SMS,.f8e4⟩,am,ak,bk,bn,cm,cn⟩
    change run (backend P) (inner p) (current.setReg "ki" .nat [] (fun _ => block))=
      some (bodyResult P p initial current tile block)
    have h : InnerInvariant P p initial tile block current := h
    simp [p,inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,integralFloorDiv,integralMod,backend,matrix,h.offAM,h.offBN,h.offK,h.acc,
      h.memory,bodyResult,accumulation,aValues,bValues,aOffset,bOffset,readCell,floatValues,dtype,
      Problem.k,clamp,Broadcast.leftIndex,Broadcast.rightIndex,RealMath.signed_tail_mask]
    try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    try simp only [← Int.natCast_mul,← Int.natCast_add,int_ite_nat,Int.ofNat_lt,Int.toNat_natCast]
    repeat' apply update_congr
    all_goals first
    | rfl
    | funext i
      try simp [p,aValues,bValues,aOffset,bOffset,readCell,clamp,Problem.k,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        -Nat.cast_mul,-Nat.cast_add,-Nat.cast_ite,
        ← Int.natCast_mul,← Int.natCast_add,int_ite_nat,masked_sum_nat]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
