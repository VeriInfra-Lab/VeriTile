import bench.tritonbench_g.matmul_persistent_triton.Inner
import bench.tritonbench_g.matmul_persistent_triton.Registers

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem fold_congr {ι β : Type} {xs : List ι} {f g : β → ι → β} {s t : β}
    (hf : f=g) (hs : s=t) : xs.foldl f s=xs.foldl g t := by
  cases hf; cases hs; rfl

private theorem update_congr {s t : State α} {n : RegName} {d : TileDType} {sh : TileShape}
    {v w : Values α d sh} (same : s=t) (values : v=w) : s.setReg n d sh v=t.setReg n d sh w := by
  cases same; cases values; rfl

def outputValue (P : Source.Arithmetic α α) (p : Parameters) (initial : State α) (tile : Nat) (i : CIndex) : Cell α :=
  pack p.problem.profile (P.cast p.problem.profile (accumulation P p initial tile p.problem.kTiles i.1 i.2.1))

def storeRegisters (P : Source.Arithmetic α α) (p : Parameters) (initial current : State α) (tile : Nat) : State α :=
  let s := current.setReg "c_ptrs" .ptr [128,256] (fun i => ("C",cOffset p tile i))
  let s := s.setReg "c_mask" .bool [128,256] (fun i => decide (active p tile i))
  s.setReg "c" (dtype p.problem.profile) [128,256]
    (floatValues p.problem.profile (fun i => P.cast p.problem.profile (accumulation P p initial tile p.problem.kTiles i.1 i.2.1)))

def suffixResult (P : Source.Arithmetic α α) (p : Parameters) (initial current : State α) (tile : Nat) : State α :=
  (TileShape.allIndices [128,256]).foldl
    (fun s i => if active p tile i then s.write "C" (cOffset p tile i) (outputValue P p initial tile i) else s)
    (storeRegisters P p initial current tile)

def written (P : Source.Arithmetic α α) (p : Parameters) (initial : State α) (tile : Nat) : State α :=
  (TileShape.allIndices [128,256]).foldl
    (fun s i => if active p tile i then s.write "C" (cOffset p tile i) (outputValue P p initial tile i) else s) initial

theorem suffixMemory (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (tile : Nat) (h : InnerInvariant P p initial tile p.problem.kTiles current) :
    (suffixResult P p initial current tile).mem=(written P p initial tile).mem := by
  simp [suffixResult,written,masked_scatter_memory,storeRegisters,h.memory]

theorem suffixExecution (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (tile : Nat) (h : InnerInvariant P p initial tile p.problem.kTiles current) :
    run (backend P) (tileSuffix p) current=some (suffixResult P p initial current tile) := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile
  case fp16 =>
    let p : Parameters := ⟨⟨M,N,K,SMS,.fp16⟩,am,ak,bk,bn,cm,cn⟩
    change run (backend P) (tileSuffix p) current=some (suffixResult P p initial current tile)
    have h : InnerInvariant P p initial tile p.problem.kTiles current := h
    simp [p,tileSuffix,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,backend,h.offAM,h.offBN,h.offCM,h.offCN,h.acc,
      suffixResult,storeRegisters,outputValue,cOffset,active,floatValues,dtype,pack,
      Broadcast.leftIndex,Broadcast.rightIndex,-List.foldl_cons,-List.foldl_nil]
    try simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
    apply fold_congr
    · funext s i
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    · repeat' apply update_congr
      all_goals first | rfl | funext i; simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  case f8e4 =>
    let p : Parameters := ⟨⟨M,N,K,SMS,.f8e4⟩,am,ak,bk,bn,cm,cn⟩
    change run (backend P) (tileSuffix p) current=some (suffixResult P p initial current tile)
    have h : InnerInvariant P p initial tile p.problem.kTiles current := h
    simp [p,tileSuffix,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,backend,h.offAM,h.offBN,h.offCM,h.offCN,h.acc,
      suffixResult,storeRegisters,outputValue,cOffset,active,floatValues,dtype,pack,
      Broadcast.leftIndex,Broadcast.rightIndex,-List.foldl_cons,-List.foldl_nil]
    try simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
    apply fold_congr
    · funext s i
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    · repeat' apply update_congr
      all_goals first | rfl | funext i; simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
