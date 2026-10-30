import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Model
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.NativeStaging
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly.Specification
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def query (M : Algebra α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head : Nat) (i : TileIndex [128,128]) : α :=
  M.fp32Load (if block*128+i.1.val<p.length ∧ i.2.1.val<96 then
    read p.Q (offset p.q batch head (block*128+i.1.val) i.2.1.val) else zero M)
def queryScale (M : Algebra α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head : Nat) : α :=
  M.fp32Load (read p.QS ((batch*p.heads+head)*((p.length+127)/128)+block))
def key (M : Algebra α) (p : Parameters) (read : RegionName→Nat→α)
    (batch head row : Nat) (i : TileIndex [128,64]) : α :=
  M.fp32Load (if i.2.1.val<p.length-row ∧ i.1.val<96 then
    read p.K (offset p.k batch head (row+i.2.1.val) i.1.val) else zero M)
def keyScale (M : Algebra α) (p : Parameters) (read : RegionName→Nat→α)
    (batch head row : Nat) : α :=
  M.fp32Load (read p.KS ((batch*p.heads+head)*((p.length+63)/64)+row/64))
def value (M : Algebra α) (p : Parameters) (read : RegionName→Nat→α)
    (batch head row : Nat) (i : TileIndex [64,128]) : α :=
  M.cast (some .fp32) .real .fp16
    (M.fp32Load (if i.1.val<p.length-row ∧ i.2.1.val<96 then
      read p.V (offset p.v batch head (row+i.1.val) i.2.1.val) else zero M))
def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (i : TileIndex [128,64]) : α :=
  if i.2.1.val<p.length-row then
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.cast (some .fp32) .real .real
          (dot.apply (some .fp32) .real .real .real {} (batch:=[])
            (query M p read block batch head) (key M p read batch head row) i))
        (queryScale M p read block batch head)) (keyScale M p read batch head row)
  else M.negInf
def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (a : Acc α) (i : Fin 128) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by decide⟩ Bool.false (scores M dot p read block batch head row) (i,()))
def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (a : Acc α) (i : TileIndex [128,64]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores M dot p read block batch head row i)
      (maximum M dot p read block batch head row a i.1))
def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (a : Acc α) (i : Fin 128) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (a.maximum i) (maximum M dot p read block batch head row a i))
def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (a : Acc α) (i : TileIndex [128,128]) : α :=
  M.cast (some .fp32) .fp16 .real
    (dot.apply (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16} (batch:=[])
      (fun j=>M.cast (some .fp32) .real .fp16 (probability M dot p read block batch head row a j))
      (value M p read batch head row) i)
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head row : Nat) (a : Acc α) : Acc α :=
  ⟨maximum M dot p read block batch head row a,
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p read block batch head row a i))
     (M.reduceSum (some .fp32) ⟨1,by decide⟩ Bool.false (probability M dot p read block batch head row a) (i,())),
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.value i) (rescale M dot p read block batch head row a i.1))
     (contribution M dot p read block batch head row a i)⟩
def initial (M : Algebra α) : Acc α :=
  ⟨fun _=>M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _=>M.binary (some .fp32) .real .add (zero M) (M.literal (some .fp32) .real 1),fun _=>zero M⟩
/-- State-free native recurrence. It retains the l=1 seed, all 128 padded
feature lanes, exact masks/casts, ordered scales and every live online update. -/
def algorithm (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName→Nat→α)
    (block batch head : Nat) (i : TileIndex [128,128]) : α :=
  let final:=foldRange (update M dot p read block batch head) 0 p.length 64 (initial M)
  M.cast (some .fp32) .real .bf16
    (M.binary (some .fp32) .real .div (final.value i) (final.normalizer i.1))
def nativeReader (c : Call) (words : Nat→BitVec 32) : RegionName→Nat→Native.Expr :=
  fun r off _=>.ready (.bits (caller c words r off))
def nativeTarget (c : Call) (words : Nat→BitVec 32) (block batch head : Nat)
    (i : TileIndex [128,128]) : Native.Graph :=
  Native.lower (algorithm Native.algebra Native.dotModel (parameters c) (nativeReader c words)
    block batch head i (Native.physicalOwner i.2.1.val))

#specNonCircular nativeTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly.Specification
