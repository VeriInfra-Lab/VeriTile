import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealTyped
import bench.tritonbench_g.var_len_copy.Proofs.Legacy
import bench.tritonbench_g.var_len_copy.Proofs.RealTyped

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.VarLenCopy
open VeriTile.Triton
open scoped VeriTile.Triton.Meta3MaskedTileKernelIO₁
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false



section IOFace

/-- **The headline on the IO surface** for `var_len_copy.py`'s
`var_len_copy_kernel_triton`, one-chunk slice: for every disjoint flat placement
of the five buffers, every program id whose metadata cells and active data lanes
are in bounds, **whatever** the three metadata scalars say, and every launch state
whose source window holds `xs`, the translated pointer kernel terminates, every
active destination lane holds the copied value `xs i`, and every other memory cell
is unchanged.

The three scalars are universally quantified and pinned by the launch state, so
the mask (`lane < length`) and both window bases (`old_start` / `new_start`) are
value-dependent — the shape the metadata skin exists for.

Dimension-general in `BLOCK_SIZE` and the chunk index, with **no**
side-condition: the destination window is `base + lane`, so the one-chunk
readback's output-injectivity precondition is discharged by `destOffset_inj`
rather than assumed. -/
theorem var_len_copy_one_chunk_io_correctness
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    varLenOneChunkIO old_a_start old_a_len old_a_location new_a_start
        new_a_location chunk BLOCK_SIZE
      ⊨ fun _pid _len _olds _news xs i => xs i := by
  refine Meta3MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact var_len_one_chunk_flattenOk old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE
  · intro bounds s len olds news hm1 hm2 hm3 hb1 hb2 hb3 hsrc hdst
    -- substituting the three pins turns the metadata scalars back into the
    -- `readMemValue` forms the per-slice lemma is phrased in, and `BlockState.pid`
    -- is reducibly `pids 0`, so the two phrasings are the same term
    subst hm1
    subst hm2
    subst hm3
    exact var_len_one_chunk_traceSafe old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE bounds s hb1 hb2 hb3
      (fun i hact => hsrc (i, PUnit.unit) hact)
      (fun i hact => hdst (i, PUnit.unit) hact)
  · intro s₀ len olds news xs hm1 hm2 hm3 hx
    exact var_len_one_chunk_region_run old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE s₀ len olds news hm1 hm2 hm3 xs
      (fun i hi => hx i hi)

end IOFace

end VeriTile.Bench.TritonBenchG.VarLenCopy

namespace VeriTile.Bench.TritonBenchG.VarLenCopy
open VeriTile Triton TypedSource FullReal
open scoped VeriTile.Spec

/-- Complete floating flat-copy conclusion: successful source execution,
actual-span frame, and initial-source readback. -/
def floatingFlatCopyContract (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) (length : Int)
    (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n))

/-- Full signed-loop IO and frame for either source version. The oracle uses
initial source cells and the actual overcopy count, with no output premises. -/
specification typed_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth v
      A s os ol ns src dst B length oldStart newStart) := by
  exact FullReal.typed_flat_io input output v A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

specification original_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth Bool.false
      A s os ol ns src dst B length oldStart newStart) := by
  exact FullReal.typed_flat_io input output Bool.false A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

specification candidate_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth Bool.true
      A s os ol ns src dst B length oldStart newStart) := by
  exact FullReal.typed_flat_io input output Bool.true A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VarLenCopy

namespace VeriTile.Bench.TritonBenchG.VarLenCopy
open VeriTile Triton FullReal
open scoped VeriTile.Spec

/-- Complete homogeneous discrete flat-copy conclusion, with exact typed
readback and the same full raw-cell frame as the floating contract. -/
def discreteFlatCopyContract (storage : DiscreteSource.Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (DiscreteSource.kernel storage v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMemValue storage.kind.dtype A.flat (A.addr dst (newStart + n)) = s.readMemValue storage.kind.dtype src (oldStart + n))

/-- Homogeneous integer and boolean payloads retain their exact typed carrier. -/
specification discrete_typed_real_correct (storage : DiscreteSource.Storage)
    (v : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage v A s os ol ns src dst B
      length oldStart newStart) := by
  exact DiscreteReal.typed_flat_io storage v A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

specification discrete_original_real_correct (storage : DiscreteSource.Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage Bool.false A s os ol ns src dst B
      length oldStart newStart) := by
  exact DiscreteReal.typed_flat_io storage Bool.false A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

specification discrete_candidate_real_correct (storage : DiscreteSource.Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage Bool.true A s os ol ns src dst B
      length oldStart newStart) := by
  exact DiscreteReal.typed_flat_io storage Bool.true A hd hcov s os ol ns src dst B length oldStart newStart
    hB hsd hL hO hN hLb hOb hNb hsrc hdst hdr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VarLenCopy
