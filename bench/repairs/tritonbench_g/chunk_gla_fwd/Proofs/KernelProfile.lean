import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeRuntime

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.KernelProfile
open VeriTile Triton FP.Structural NativeReduction NativeStaging
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

/-- Finite local SM80 reduction-layout metadata. The selected compiler can
vectorize one, two or four contiguous features per thread. Pointer alignment
and K's stride divisibility select this metadata; neither input values nor a
reduction/program result are fields. All four source autotune warp choices
and all three vector widths are covered by the execution theorem. -/
structure Profile where
  warps : Fin 4
  vectorBits : Fin 3

def lowBits (e : Fin 9) (p : Profile) := min e.val p.vectorBits.val
def laneBits (e : Fin 9) (p : Profile) := min 5 (e.val-lowBits e p)
def warpBits (e : Fin 9) (p : Profile) := min p.warps.val (e.val-lowBits e p-laneBits e p)
def highBits (e : Fin 9) (p : Profile) := e.val-lowBits e p-laneBits e p-warpBits e p

/-- The physical blocked layout has contiguous register bits, then lane bits,
then warp bits, then remaining register bits. Its register fold and descending
XOR butterfly keep operand order and all materialization boundaries explicit. -/
def plan (e : Fin 9) (p : Profile) : Tree :=
  let vec := 2^lowBits e p
  let lanes := 2^laneBits e p
  let warps := 2^warpBits e p
  let regs := 2^(lowBits e p+highBits e p)
  let within := fun warp => NativeReduction.butterfly
    (fun lane => NativeReduction.registerTree ((List.range regs).map (fun reg =>
      reg/vec*(warps*lanes*vec)+warp*(lanes*vec)+lane*vec+reg%vec))) (laneBits e p) 1 0
  if warpBits e p=0 then within 0
  else NativeReduction.butterfly (fun warp => .move .shared (within warp)) (warpBits e p) 1 0

theorem coverage (e : Fin 9) (p : Profile) :
    (plan e p).leaves.Perm (List.range (2^e.val)) := by
  rcases p with ⟨w,v⟩
  fin_cases e <;> fin_cases w <;> fin_cases v <;> decide

theorem no_duplicate (e : Fin 9) (p : Profile) : (plan e p).leaves.Nodup :=
  (coverage e p).nodup_iff.mpr List.nodup_range

def planFor (n : Nat) (p : Profile) : Tree :=
  match n with
  | 1 => plan ⟨0,by decide⟩ p
  | 2 => plan ⟨1,by decide⟩ p
  | 4 => plan ⟨2,by decide⟩ p
  | 8 => plan ⟨3,by decide⟩ p
  | 16 => plan ⟨4,by decide⟩ p
  | 32 => plan ⟨5,by decide⟩ p
  | 64 => plan ⟨6,by decide⟩ p
  | 128 => plan ⟨7,by decide⟩ p
  | 256 => plan ⟨8,by decide⟩ p
  | n => registerTree (List.range n)

def reduction (p : Profile) (precision : Option ComputeDType) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  (planFor (TileShape.axisDim shape ax) p).value (.literal precision .real 0)
    (fun k => if h : k<TileShape.axisDim shape ax then NativeStaging.reductionInputs ax keep xs i ⟨k,h⟩
      else .literal precision .real 0) Expr.move (Expr.binary precision .real .add)

def algebra (p : Profile) (scale : BitVec 32) : Algebra Expr :=
  { NativeRuntime.algebra p.warps scale with reduceSum := reduction p }

/-- The actually observed aligned K=64 reduction uses four register features
and sixteen lanes. In particular it does not use the old 32-lane/2-warp tree. -/
theorem aligned64 :
    lowBits ⟨6,by decide⟩ ⟨⟨2,by decide⟩,⟨2,by decide⟩⟩=2 ∧
    laneBits ⟨6,by decide⟩ ⟨⟨2,by decide⟩,⟨2,by decide⟩⟩=4 ∧
    warpBits ⟨6,by decide⟩ ⟨⟨2,by decide⟩,⟨2,by decide⟩⟩=0 := by decide

#axiomsClean coverage
#axiomsClean no_duplicate
#axiomsClean aligned64
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.KernelProfile
