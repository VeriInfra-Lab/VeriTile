import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Memory

/-!
Erased-real whole-program memory model. The arithmetic recurrences follow the
source's normalized online forward, preprocessing, and triangular backward.
Scatter performs every logical store and preserves the rest of memory. This
model is deliberately distinct from the pending native floating-point graph;
no FP-to-Real numerical equality is asserted here.
-/
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001
namespace RealProgram
open Memory
open scoped BigOperators
noncomputable section

attribute [simp] Memory.read_writeTensor Memory.read_writeTensor_other
  Memory.read_writeRows Memory.readRows_writeTensor_other
  Memory.readTensor_writeRows_other Memory.readRows_writeRows_other

def forwardY (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :=
  Math.sourceForwardOutput (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) scale

def forwardL (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :=
  Math.sourceForwardL (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) scale

def forwardM (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :=
  Math.sourceForwardM (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) scale

def forward (c : Shape) (s : Memory ℝ) (scale : ℝ) : Memory ℝ :=
  let so := writeTensor c s .o (fun p => forwardY c s scale p.1 p.2.1 p.2.2)
  let sl := writeRows c so .l (fun p => forwardL c s scale p.1 p.2)
  writeRows c sl .m (fun p => forwardM c s scale p.1 p.2)

/-- Preprocess reads the current O/L memory left by the forward grid. -/
def newDO (c : Shape) (s : Memory ℝ) (h : Fin c.groups) : Math.Tensor c.rows c.dim :=
  Math.preprocessDO (readTensor c s .g h) (readRows c s .l h)
def delta (c : Shape) (s : Memory ℝ) (h : Fin c.groups) : Fin c.rows → ℝ :=
  Math.preprocessDelta (readTensor c s .o h) (readTensor c s .g h) (readRows c s .l h)
def preprocess (c : Shape) (s : Memory ℝ) : Memory ℝ :=
  let sn := writeTensor c s .newDO (fun p => newDO c s p.1 p.2.1 p.2.2)
  writeRows c sn .delta (fun p => delta c s p.1 p.2)

/-- Backward score recomputation uses saved M; DP reads current NewDO and
Delta. These are actual stage connections, not independently supplied arrays. -/
def probability (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) :=
  Math.sourceP (readTensor c s .q h) (readTensor c s .k h) scale (readRows c s .m h)
def ds (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups)
    (i j : Fin c.rows) : ℝ :=
  probability c s scale h i j *
    ((∑ e, readTensor c s .newDO h i e * readTensor c s .v h j e) -
      readRows c s .delta h i) * scale

def dq (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) : Math.Tensor c.rows c.dim :=
  fun i e => Math.lowerTiledSum c i (fun j => ds c s scale h i j * readTensor c s .k h j e)
def dk (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) : Math.Tensor c.rows c.dim :=
  fun j e => Math.upperTiledSum c j (fun i => ds c s scale h i j * readTensor c s .q h i e)
def dv (c : Shape) (s : Memory ℝ) (scale : ℝ) (h : Fin c.groups) : Math.Tensor c.rows c.dim :=
  fun j e => Math.upperTiledSum c j (fun i =>
    probability c s scale h i j * readTensor c s .newDO h i e)

def backward (c : Shape) (s : Memory ℝ) (scale : ℝ) : Memory ℝ :=
  let sq := writeTensor c s .dq (fun p => dq c s scale p.1 p.2.1 p.2.2)
  let sk := writeTensor c sq .dk (fun p => dk c s scale p.1 p.2.1 p.2.2)
  writeTensor c sk .dv (fun p => dv c s scale p.1 p.2.1 p.2.2)

def run (c : Shape) (s : Memory ℝ) (scale : ℝ) : Memory ℝ :=
  backward c (preprocess c (forward c s scale)) scale

/-- Zero-initialized repeated DQ updates are a prefix sum. This local
invariant is the erased-real meaning of each source DQ read/add/write. -/
def dqPrefix (term : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => dqPrefix term k + term k

theorem dqPrefix_eq_sum (term : Nat → ℝ) (k : Nat) :
    dqPrefix term k = ∑ j ∈ Finset.range k, term j := by
  induction k with
  | zero => simp [dqPrefix]
  | succ k ih => simpa only [dqPrefix,Finset.sum_range_succ,ih]

#specNonCircular Math.attention avoiding [forward,preprocess,backward,run]
#specNonCircular Math.gradientQ avoiding [forward,preprocess,backward,run]
#specNonCircular Math.gradientK avoiding [forward,preprocess,backward,run]
#specNonCircular Math.gradientV avoiding [forward,preprocess,backward,run]
#axiomsClean dqPrefix_eq_sum
end
end RealProgram
end VeriTile.Bench.Repairs.TritonAttention.Whole001
