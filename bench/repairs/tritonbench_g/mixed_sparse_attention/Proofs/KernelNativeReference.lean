import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.NativeScale

/-! A native recurrence on initial input cells. Its definition uses no source
statement evaluator, source update/finalValues, or returned output memory. -/
namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelNativeReference
open VeriTile Triton FP.Structural
open Whole002 Native Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def length (c : Config) (s : State Expr) := (s.mem (Region.cast c.seq) (s.pids 1/c.H)).read .nat
def row (c : Config) (s : State Expr) := s.pids 0*c.BM
def slot (c : Config) (s : State Expr) := s.pids 1*c.NR+s.pids 0
def blocks (c : Config) (s : State Expr) := (s.mem (Region.cast c.blocks) (slot c s)).read .nat
def columns (c : Config) (s : State Expr) := (s.mem (Region.cast c.counts) (slot c s)).read .nat
def qbase (c : Config) (s : State Expr) := s.pids 1/c.H*c.sqz+s.pids 1%c.H*c.sqh
def kvbase (c : Config) (s : State Expr) := s.pids 1/c.H*c.skz+s.pids 1%c.H*c.skh
def selected (c : Config) (s : State Expr) (phase : Bool) (start : Nat) (j : Fin c.BN) : Nat :=
  if phase then
    if start+j.val<columns c s ∧ start<columns c s then
      (s.mem (Region.cast c.columns) (slot c s*c.NV+start+j.val)).read .nat else 0
  else (if start<blocks c s then (s.mem (Region.cast c.offsets) (slot c s*c.NS+start)).read .nat else 0)+j.val
def present (c : Config) (s : State Expr) (phase : Bool) (start : Nat) (j : Fin c.BN) : Prop :=
  if phase then start+j.val<columns c s ∧ start<columns c s
  else selected c s Bool.false start j<length c s ∧ start<blocks c s
instance (c : Config) (s : State Expr) (phase : Bool) (n : Nat) (j : Fin c.BN) :
    Decidable (present c s phase n j) := by unfold present;infer_instance

def query (rt : Runtime) (c : Config) (s : State Expr) (i : TileIndex [c.BM,c.BD]) : Expr :=
  (algebra rt).cast none .real c.storage.float ((algebra rt).binary (some .fp32) .real .mul
    (if row c s+i.1.val<length c s then
      (s.mem c.q (qbase c s+(row c s+i.1.val)*c.sqm+i.2.1.val*c.sqk)).read .real
     else (algebra rt).literal (some .fp32) .real 0.0)
    ((algebra rt).binary (some .fp32) .real .mul ((algebra rt).literal (some .fp32) .real c.scale)
      ((algebra rt).literal (some .fp32) .real 1.44269504)))
def key (rt : Runtime) (c : Config) (s : State Expr) (ph : Bool) (n : Nat) (i : TileIndex [c.BD,c.BN]) : Expr :=
  (algebra rt).cast none .real c.storage.float
    (if present c s ph n i.2.1 then
      (s.mem c.k (kvbase c s+i.1.val*c.skk+selected c s ph n i.2.1*c.skn)).read .real
     else (algebra rt).literal (some .fp32) .real 0.0)
def value (rt : Runtime) (c : Config) (s : State Expr) (ph : Bool) (n : Nat) (i : TileIndex [c.BN,c.BD]) : Expr :=
  (algebra rt).cast none .real c.storage.float
    (if present c s ph n i.1 then
      (s.mem c.v (kvbase c s+i.2.1.val*c.svk+selected c s ph n i.1*c.svn)).read .real
     else (algebra rt).literal (some .fp32) .real 0.0)

def initial (rt : Runtime) (c : Config) : Accumulator Expr c.BM c.BD :=
  ⟨fun _=>(algebra rt).binary (some .fp32) .real .add ((algebra rt).literal (some .fp32) .real 0) (algebra rt).negInf,
   fun _=>(algebra rt).literal (some .fp32) .real 0,fun _=>(algebra rt).literal (some .fp32) .real 0⟩
def step (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr)
    (phase : Bool) (n : Nat) (a : Accumulator Expr c.BM c.BD) : Accumulator Expr c.BM c.BD :=
  let M:=algebra rt
  let dot:=dotModel c.storage I
  let qk:=dot.apply (some .fp32) {} (batch:=[]) (query rt c s) (key rt c s phase n)
  let score : TileIndex [c.BM,c.BN] → Expr := fun ij=>
    M.binary (some .fp32) .real .add
      (if row c s+ij.1.val<length c s ∧
        (if phase then present c s Bool.true n ij.2.1
         else present c s Bool.false n ij.2.1 ∧ selected c s Bool.false n ij.2.1≤row c s+ij.1.val)
       then M.literal (some .fp32) .real 0 else M.negInf)
      (M.cast (some .fp32) .fp32 .real (qk ij))
  let peak:=M.reduceMax (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false score
  let maximum : TileIndex [c.BM] → Expr := fun i=>M.binary (some .fp32) .real .max (a.maximum i) (peak i)
  let shift : TileIndex [c.BM] → Expr := fun i=>if dot.equal (maximum i) M.negInf then M.literal (some .fp32) .real 0 else maximum i
  let gain : TileIndex [c.BM] → Expr := fun i=>M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (a.maximum i) (shift i))
  let weights : TileIndex [c.BM,c.BN] → Expr := fun ij=>M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (score ij) (shift (ij.1,())))
  let contribution:=dot.apply (some .fp32) {} (batch:=[])
    (fun ij=>M.cast (some .fp32) .real c.storage.float (weights ij)) (value rt c s phase n)
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
def accumulate (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr) :=
  foldRange (step rt I c s Bool.true) 0 (columns c s) c.BN
    (foldRange (step rt I c s Bool.false) 0 (blocks c s) 1 (initial rt c))
def answer (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr)
    (ij : TileIndex [c.BM,c.BD]) : Graph :=
  let a:=accumulate rt I c s
  let M:=algebra rt
  lower ((M.cast (some .fp32) c.storage.float .real
    (M.cast (some .fp32) .real c.storage.float
      (if (dotModel c.storage I).equal (a.normalizer (ij.1,())) (M.literal (some .fp32) .real 0)
       then M.literal (some .fp32) .real 0
       else M.binary (some .fp32) .real .div (a.output ij) (a.normalizer (ij.1,()))))) (physicalOwner ij.2.1.val))

theorem query_eq (rt : Runtime) (c : Config) (s : State Expr) :
    query rt c s=Source.queryValues (algebra rt) c s := rfl
theorem step_eq (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr)
    (ph : Bool) (n : Nat) (a : Accumulator Expr c.BM c.BD) :
    step rt I c s ph n a=Source.update (algebra rt) (dotModel c.storage I) c s ph n a := rfl
theorem accumulate_eq (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr) :
    accumulate rt I c s=Source.finalValues (algebra rt) (dotModel c.storage I) c s := rfl
theorem answer_eq (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (s : State Expr)
    (ij : TileIndex [c.BM,c.BD]) :
    answer rt I c s ij=lower ((Whole.answer (algebra rt) (dotModel c.storage I) c s ij) (physicalOwner ij.2.1.val)) := rfl

#axiomsClean query_eq
#axiomsClean step_eq
#axiomsClean accumulate_eq
#axiomsClean answer_eq
#specNonCircular answer avoiding [Source.finalValues,Source.update,Source.outputValue,
  Source.queryValues,Source.keyValues,Source.valueValues,FP.Structural.exec,Whole.runPrograms]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelNativeReference
