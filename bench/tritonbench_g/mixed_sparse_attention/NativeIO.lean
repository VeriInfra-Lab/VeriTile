import bench.tritonbench_g.mixed_sparse_attention.IO
import bench.tritonbench_g.mixed_sparse_attention.Proofs.NativeScale

namespace VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
open Classical

/-- Original FP16/BF16 storage selection, FP32 computation and complete supplied
program order. No optimized second kernel or native-to-Real claim is introduced. -/
def io (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (work : List Program) :=
  KernelIO.io (algebra rt) (dotModel c.storage I) c work

def query (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c) (p : Program)
    (i : TileIndex [c.BM,c.BD]) : Expr :=
  (algebra rt).cast none .real c.storage.float ((algebra rt).binary (some .fp32) .real .mul
    (if KernelIO.row c p+i.1.val<KernelIO.sequence c xs p then
      KernelIO.query c xs (KernelIO.qbase c p+(KernelIO.row c p+i.1.val)*c.sqm+i.2.1.val*c.sqk)
     else (algebra rt).literal (some .fp32) .real 0.0)
    ((algebra rt).binary (some .fp32) .real .mul ((algebra rt).literal (some .fp32) .real c.scale)
      ((algebra rt).literal (some .fp32) .real 1.44269504)))
def key (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c) (p : Program)
    (ph : Bool) (n : Nat) (i : TileIndex [c.BD,c.BN]) : Expr :=
  (algebra rt).cast none .real c.storage.float
    (if KernelIO.keyActive c xs p ph n i.2.1 then
      KernelIO.key c xs (KernelIO.kvbase c p+i.1.val*c.skk+KernelIO.selected c xs p ph n i.2.1*c.skn)
     else (algebra rt).literal (some .fp32) .real 0.0)
def value (rt : Runtime) (c : Config) (xs : KernelIO.Values Expr c) (p : Program)
    (ph : Bool) (n : Nat) (i : TileIndex [c.BN,c.BD]) : Expr :=
  (algebra rt).cast none .real c.storage.float
    (if KernelIO.keyActive c xs p ph n i.1 then
      KernelIO.value c xs (KernelIO.kvbase c p+i.2.1.val*c.svk+KernelIO.selected c xs p ph n i.1*c.svn)
     else (algebra rt).literal (some .fp32) .real 0.0)

def initial (rt : Runtime) (c : Config) : Accumulator Expr c.BM c.BD :=
  ⟨fun _=>(algebra rt).binary (some .fp32) .real .add ((algebra rt).literal (some .fp32) .real 0) (algebra rt).negInf,
   fun _=>(algebra rt).literal (some .fp32) .real 0,fun _=>(algebra rt).literal (some .fp32) .real 0⟩
def step (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (xs : KernelIO.Values Expr c)
    (p : Program) (phase : Bool) (n : Nat) (a : Accumulator Expr c.BM c.BD) : Accumulator Expr c.BM c.BD :=
  let M:=algebra rt
  let dot:=dotModel c.storage I
  let qk:=dot.apply (some .fp32) {} (batch:=[]) (query rt c xs p) (key rt c xs p phase n)
  let score : TileIndex [c.BM,c.BN] → Expr := fun ij=>
    M.binary (some .fp32) .real .add
      (if KernelIO.row c p+ij.1.val<KernelIO.sequence c xs p ∧
        (if phase then KernelIO.keyActive c xs p Bool.true n ij.2.1
         else KernelIO.keyActive c xs p Bool.false n ij.2.1 ∧
           KernelIO.selected c xs p Bool.false n ij.2.1≤KernelIO.row c p+ij.1.val)
       then M.literal (some .fp32) .real 0 else M.negInf)
      (M.cast (some .fp32) .fp32 .real (qk ij))
  let peak:=M.reduceMax (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false score
  let maximum : TileIndex [c.BM] → Expr := fun i=>M.binary (some .fp32) .real .max (a.maximum i) (peak i)
  let shift : TileIndex [c.BM] → Expr := fun i=>if dot.equal (maximum i) M.negInf then M.literal (some .fp32) .real 0 else maximum i
  let gain : TileIndex [c.BM] → Expr := fun i=>M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (a.maximum i) (shift i))
  let weights : TileIndex [c.BM,c.BN] → Expr := fun ij=>M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (score ij) (shift (ij.1,())))
  let contribution:=dot.apply (some .fp32) {} (batch:=[])
    (fun ij=>M.cast (some .fp32) .real c.storage.float (weights ij)) (value rt c xs p phase n)
  let mass:=M.reduceSum (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false weights
  { maximum:=maximum
    normalizer:=fun i=>M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (a.normalizer i) (gain i)) (mass i)
    output:=fun ij=>M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (a.output ij)
        (M.binary (some .fp32) .real .add
          (M.binary (some .fp32) .real .mul (a.normalizer (ij.1,())) (M.literal (some .fp32) .real 0))
          (gain (ij.1,()))))
      (M.cast (some .fp32) .fp32 .real (contribution ij)) }
def accumulate (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (xs : KernelIO.Values Expr c) (p : Program) :=
  foldRange (step rt I c xs p Bool.true) 0 (KernelIO.columnCount c xs p) c.BN
    (foldRange (step rt I c xs p Bool.false) 0 (KernelIO.blockCount c xs p) 1 (initial rt c))
def expression (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (xs : KernelIO.Values Expr c) (p : Program) (ij : TileIndex [c.BM,c.BD]) : Expr :=
  let a:=accumulate rt I c xs p
  let M:=algebra rt
  M.cast (some .fp32) c.storage.float .real
    (M.cast (some .fp32) .real c.storage.float
      (if (dotModel c.storage I).equal (a.normalizer (ij.1,())) (M.literal (some .fp32) .real 0)
       then M.literal (some .fp32) .real 0
       else M.binary (some .fp32) .real .div (a.output ij) (a.normalizer (ij.1,()))))

def observe (c : Config) (I : CompareISA (BitVec 32)) :
    (o : Fin (KernelIO.outputs c).length) → TileIndex (KernelIO.outputs c)[o].shape → Expr → BitVec 32
  | ⟨0,_⟩,i,x=>(lower (x (physicalOwner i.2.2.1.val))).eval I.toISA

def formula (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (hbm : 0<c.BM)
    (xs : KernelIO.Values Expr c) : ChannelValues (BitVec 32) (KernelIO.outputs c) :=
  fun
  | ⟨0,_⟩,i=>observe c I (0 : Fin 1) i (expression rt I c xs
    (i.2.1.val/c.BM,i.1.val,0) (lane c hbm i.2.1.val i.2.2.1.val i.2.2.1.isLt))

#specNonCircular formula avoiding [FP.Structural.exec,runSchedule,Whole.runPrograms,
  Source.finalValues,Source.update,State.mem]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
