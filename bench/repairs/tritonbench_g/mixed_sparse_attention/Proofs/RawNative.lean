import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelNativeReference

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RawNative
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid
noncomputable section

/-- Complete raw native source grid versus its independent ordered reference.
All input word patterns and the original native instruction model are retained. -/
theorem kernel_native (I : Native.CompareISA (BitVec 32)) (rt : Native.Runtime)
    (c : Config) (sep : Separated c) (layout : Contiguous c) (s : State Native.Expr)
    (lengths : LengthsBound c s) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms (fun _=>Native.algebra rt) (fun _=>Native.dotModel c.storage I)
      c (grid c) work s=some final ∧
      (∀ q : Raw.Coord c,live c s q →
        Native.readNative I c (final.mem c.out (Raw.offset q)) (Raw.dim q)=
          (KernelNativeReference.answer rt I c (start (grid c) (owner c q) s)
            (rowLane c hbm q)).eval I.toISA) ∧
      (∀ r i,¬Writes c s r i → final.mem r i=s.mem r i) := by
  obtain ⟨final,he,hr⟩:=KernelGrid.source (fun _=>Native.algebra rt)
    (fun _=>Native.dotModel c.storage I) c sep layout s lengths hbm hbn work order
  refine ⟨final,he,?_,hr.frame⟩
  intro q hq
  rw [hr.output q hq]
  simp only [Native.read_native_word,KernelNativeReference.answer_eq,rowLane,lane]

#axiomsClean kernel_native
#specNonCircular KernelNativeReference.answer avoiding [Source.finalValues,Source.update,FP.Structural.exec,Whole.runPrograms]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.RawNative
