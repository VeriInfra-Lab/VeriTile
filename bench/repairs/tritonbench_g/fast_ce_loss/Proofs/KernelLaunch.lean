import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelAccess
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelLaunch
open VeriTile.Triton FP.Structural KernelSource KernelContract KernelContract.Storage
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

def blocks (vocab width : Nat) : Nat := (vocab+width-1)/width
def nextPowerOfTwo (vocab : Nat) : Nat := 2^(Nat.clog 2 vocab)
def forwardWarps (width : Nat) : Nat :=
  if 32768≤width then 32 else if 8192≤width then 16 else if 2048≤width then 8 else 4

structure ForwardGeometry (p : Parameters) (warps : Nat) : Prop where
  single : p.vocab≤65536
  width : p.block=nextPowerOfTwo p.vocab
  warps : warps=forwardWarps p.block

structure ChunkedGeometry (p : Parameters) (warps : Nat) : Prop where
  split : 65536<p.vocab
  width : p.block=65536
  count : p.chunks=blocks p.vocab 65536
  warps : warps=32

structure BackwardGeometry (p : Parameters) (count warps : Nat) : Prop where
  width : p.block=4096
  count : count=blocks p.vocab 4096
  warps : warps=8

 theorem forward_covered (p : Parameters) (geometry : ForwardGeometry p warps) : p.vocab≤p.block := by
  rw [geometry.width,nextPowerOfTwo]
  exact Nat.le_pow_clog (by decide) _

/-- These constructors derive flat Tensor index laws from observed metadata.
Singleton stride entries stay unconstrained when no coordinate uses them. -/
 theorem flat_vector (t : Tensor) (region : RegionName) (count stride : Nat)
    (view : VectorView t region count stride) (used : 1<count → stride=1) : FlatView t region count := by
  refine ⟨view.regionEq,?_,?_⟩
  · simp [Tensor.numel,view.shape]
  · intro i hi
    rw [vector_index t region count stride i view hi]
    by_cases many : 1<count
    · simp [used many]
    · have zero : i=0 := by omega
      simp [zero]

 theorem flat_matrix (t : Tensor) (region : RegionName) (rows cols stride : Nat)
    (view : MatrixView t region rows cols stride) (used : 1<rows → stride=cols) :
    FlatView t region (rows*cols) := by
  refine ⟨view.regionEq,?_,?_⟩
  · simp [Tensor.numel,view.shape]
  · intro i hi
    have positive : 0<cols := by nlinarith
    have hr : i/cols<rows := (Nat.div_lt_iff_lt_mul positive).mpr hi
    have hc : i%cols<cols := Nat.mod_lt i positive
    have decompose : i/cols*cols+i%cols=i := by simpa [Nat.mul_comm] using Nat.div_add_mod i cols
    have result := matrix_index t region rows cols stride (i/cols) (i%cols) view hr hc
    rw [decompose] at result
    rw [result]
    by_cases many : 1<rows
    · rw [used many]
      omega
    · have zero : i/cols=0 := Nat.lt_one_iff.mp (lt_of_lt_of_le hr (Nat.le_of_not_gt many))
      rw [zero] at decompose ⊢
      simp only [Nat.zero_mul,Nat.add_zero,Nat.zero_add] at decompose ⊢
      rw [decompose]

 theorem column_signed (block width lane : Nat) (hb : block≤65535) (hw : width≤65536)
    (hl : lane<width) :
    (KernelAccess.column64 block width lane).toInt=(block*width+lane : Nat) := by
  rw [KernelAccess.column_bits block width lane (by omega)]
  apply VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native.nat64_signed
  have product := Nat.mul_le_mul hb hw
  norm_num at product ⊢
  omega

 theorem column_mask (vocab block width lane : Nat) (hv : vocab≤4096*65535)
    (hb : block≤65535) (hw : width≤65536) (hl : lane<width) :
    (KernelAccess.column64 block width lane).slt (BitVec.ofNat 64 vocab)=decide (block*width+lane<vocab) := by
  rw [BitVec.slt, column_signed block width lane hb hw hl,
    VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native.nat64_signed vocab (by norm_num at hv ⊢; omega)]
  norm_cast

#axiomsClean forward_covered
#axiomsClean flat_vector
#axiomsClean flat_matrix
#axiomsClean column_signed
#axiomsClean column_mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelLaunch
