import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeMetadata
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeReal
import bench.tritonbench_g.relu_strided_buffer.Proofs.Real
import bench.tritonbench_g.relu_strided_buffer.Proofs.CandidateReal
import VeriTile.Meta.StatementAudit

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁
open scoped VeriTile.Triton.StreamGridStrideEmitMasked2DKernelIO₁



/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, `one_tile_per_cta = true` branch: for every
disjoint flat placement of the two buffers, every program id whose active lanes
are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
`relu (xs i) = max 0 (xs i)`, and every other memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`; the only side-condition
is `0 < out0_stride0` (store-footprint injectivity — a torch stride of a
non-degenerate rank-1 buffer is ≥ 1). The strided, two-stride footprint is
carried verbatim by `MaskedTileKernelIO₁`'s address functions. -/
theorem relu_strided_buffer_one_tile_io_correctness
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) :
    reluOneTileIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => TiledActivation.relu (xs i) := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_one_tile_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s h1 h2
    exact relu_one_tile_traceSafe in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 bounds s
      (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
  · intro s₀ xs hin
    exact relu_one_tile_region_run in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 hStride s₀ xs hin


/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the grid-width-aware IO surface** for
`relu_strided_buffer.py`'s `relu_forward_kernel_rank_1`,
`one_tile_per_cta = false` (grid-stride) branch: for every disjoint flat
placement of the two buffers, every program id and **launch grid width**, and
every launch state whose per-step input windows hold `xs`, the translated
pointer kernel terminates and every write-active lane of every grid-stride step
holds `relu (xs t i) = max 0 (xs t i)`, with every other memory **cell**
unchanged.

Dimension-general in `s0`, both strides, `tile_size0` and `tiles_per_cta`, and
**universally quantified over the launch grid width** — the point of the skin:
the windows stride by `nCtas = tl.num_programs(0)`, pinned inside the relation
to `s₀.numPids 0`.

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity — a
torch stride of a non-degenerate rank-1 buffer is ≥ 1); `in0_ptr ≠ out0_ptr`
(later grid-stride steps load after earlier steps stored, and the two buffers
carry *different* strides, so an aliased pair genuinely breaks the contract —
the IO surface's placement disjointness does not supply this, since the core
allows a skin to name one region twice for in-place kernels); and the skin's
`pre` `pid₀ < nCtas` (launch legality, which also supplies the `0 < nCtas` the
loop arithmetic needs).

`R` is threaded through the whole kernel but rounds nothing — with
`outDType := .real` every per-step store is exact under `execR R`, so this is
the exact grid-stride streaming contract. That is faithful rather than vacuous
because ReLU is a *selection* (`tl.where`), so no arithmetic manufactures a real
that would need quantizing. -/
theorem relu_strided_buffer_grid_stride_io_correctnessR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr) :
    reluGridStrideIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨[R] fun _pid₀ _pid₁ _nCtas xs t i => TiledActivation.relu (xs t i) := by
  refine StreamGridStrideEmitMasked2DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact relu_grid_stride_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s xs _ _ hbr1 hbw
    exact relu_grid_stride_traceSafeR R in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 hStride bounds s hbr1 hbw
  · intro s₀ xs hpre _ hx
    simp only [reluGridStrideIO] at hpre hx ⊢
    have hGrid : 0 < s₀.numPids 0 := Nat.lt_of_le_of_lt (Nat.zero_le _) hpre
    obtain ⟨s1, hexec, hcells⟩ :=
      relu_grid_stride_run_exact in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        num_tasks tiles_per_cta tile_size0 hStride s₀
    refine ⟨s1, ?_, ?_, ?_⟩
    · rw [relu_grid_stride_execR_eq_exec R in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀]
      exact hexec
    · intro t i hact
      have hval := relu_grid_stride_exec_correct in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀ hStride hDisj
        hGrid s1 hexec t.val i t.isLt hact
      rw [BlockState.readMemAs_real]
      refine congrArg some ?_
      rw [hval]
      simp only [reluSpec, FloatDType.ofReal, RoundingModel.round_real_apply]
      rw [hx t i hact]
    · intro r o hcond
      refine hcells r o ?_
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun j i hj hi => h ⟨j, hj⟩ i hi


namespace Candidate


/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, `one_tile_per_cta = true` branch: for every
disjoint flat placement of the two buffers, every program id whose active lanes
are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
`relu (xs i) = max 0 (xs i)`, and every other memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`; the only side-condition
is `0 < out0_stride0` (store-footprint injectivity — a torch stride of a
non-degenerate rank-1 buffer is ≥ 1). The strided, two-stride footprint is
carried verbatim by `MaskedTileKernelIO₁`'s address functions. -/
theorem relu_strided_buffer_one_tile_io_correctness
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (_hB : 0 < tile_size0) :
    reluOneTileIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => TiledActivation.relu (xs i) := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_one_tile_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s h1 h2
    exact relu_one_tile_traceSafe in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 bounds s
      (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
  · intro s₀ xs hin
    exact relu_one_tile_region_run in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 hStride s₀ xs hin


/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the grid-width-aware IO surface** for
`relu_strided_buffer.py`'s `relu_forward_kernel_rank_1`,
`one_tile_per_cta = false` (grid-stride) branch: for every disjoint flat
placement of the two buffers, every program id and **launch grid width**, and
every launch state whose per-step input windows hold `xs`, the translated
pointer kernel terminates and every write-active lane of every grid-stride step
holds `relu (xs t i) = max 0 (xs t i)`, with every other memory **cell**
unchanged.

Dimension-general in `s0`, both strides, `tile_size0` and `tiles_per_cta`, and
**universally quantified over the launch grid width** — the point of the skin:
the windows stride by `nCtas = tl.num_programs(0)`, pinned inside the relation
to `s₀.numPids 0`.

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity — a
torch stride of a non-degenerate rank-1 buffer is ≥ 1); `in0_ptr ≠ out0_ptr`
(later grid-stride steps load after earlier steps stored, and the two buffers
carry *different* strides, so an aliased pair genuinely breaks the contract —
the IO surface's placement disjointness does not supply this, since the core
allows a skin to name one region twice for in-place kernels); and the skin's
`pre` `pid₀ < nCtas` (launch legality, which also supplies the `0 < nCtas` the
loop arithmetic needs).

`R` is threaded through the whole kernel but rounds nothing — with
`outDType := .real` every per-step store is exact under `execR R`, so this is
the exact grid-stride streaming contract. That is faithful rather than vacuous
because ReLU is a *selection* (`tl.where`), so no arithmetic manufactures a real
that would need quantizing. -/
theorem relu_strided_buffer_grid_stride_io_correctnessR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr)
    (_hB : 0 < tile_size0) :
    reluGridStrideIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨[R] fun _pid₀ _pid₁ _nCtas xs t i => TiledActivation.relu (xs t i) := by
  refine StreamGridStrideEmitMasked2DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact relu_grid_stride_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s xs _ _ hbr1 hbw
    exact relu_grid_stride_traceSafeR R in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 hStride bounds s hbr1 hbw
  · intro s₀ xs hpre _ hx
    simp only [reluGridStrideIO] at hpre hx ⊢
    have hGrid : 0 < s₀.numPids 0 := Nat.lt_of_le_of_lt (Nat.zero_le _) hpre
    obtain ⟨s1, hexec, hcells⟩ :=
      relu_grid_stride_run_exact in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        num_tasks tiles_per_cta tile_size0 hStride s₀
    refine ⟨s1, ?_, ?_, ?_⟩
    · rw [relu_grid_stride_execR_eq_exec R in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀]
      exact hexec
    · intro t i hact
      have hval := relu_grid_stride_exec_correct in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀ hStride hDisj
        hGrid s1 hexec t.val i t.isLt hact
      rw [BlockState.readMemAs_real]
      refine congrArg some ?_
      rw [hval]
      simp only [reluSpec, FloatDType.ofReal, RoundingModel.round_real_apply]
      rw [hx t i hact]
    · intro r o hcond
      refine hcells r o ?_
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun j i hj hi => h ⟨j, hj⟩ i hi

end Candidate
specification original_one_tile_correctness
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy)
    (_hB : 0 < B) :
    Spec.Real (reluOneTileIO x y sx sy n tasks tiles B
      ⊨ fun _pid xs i => TiledActivation.relu (xs i)) :=
  relu_strided_buffer_one_tile_io_correctness x y sx sy n tasks tiles B hs

specification original_grid_stride_correctness (R : RoundingModel)
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy) (hxy : x ≠ y)
    (_hB : 0 < B) :
    Spec.Real (reluGridStrideIO x y sx sy n tasks tiles B
      ⊨[R] fun _p₀ _p₁ _grid xs t i => TiledActivation.relu (xs t i)) :=
  relu_strided_buffer_grid_stride_io_correctnessR R x y sx sy n tasks tiles B hs hxy

specification candidate_one_tile_correctness
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy)
    (_hB : 0 < B) :
    Spec.Real (Candidate.reluOneTileIO x y sx sy n tasks tiles B
      ⊨ fun _pid xs i => TiledActivation.relu (xs i)) :=
  Candidate.relu_strided_buffer_one_tile_io_correctness x y sx sy n tasks tiles B hs _hB

specification candidate_grid_stride_correctness (R : RoundingModel)
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy) (hxy : x ≠ y)
    (_hB : 0 < B) :
    Spec.Real (Candidate.reluGridStrideIO x y sx sy n tasks tiles B
      ⊨[R] fun _p₀ _p₁ _grid xs t i => TiledActivation.relu (xs t i)) :=
  Candidate.relu_strided_buffer_grid_stride_io_correctnessR R x y sx sy n tasks tiles B hs hxy _hB

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton FP.Structural HostPlan SignedView SignedGrid
noncomputable section

/-- Complete original rank-one host call under its independent mathematical
interpretation: each logical output is max(0,x), the returned tensor is exactly
out0, and every cell outside the output footprint is preserved from entry.
Either signed stride is allowed; input stride zero is allowed. Output stride
must be nonzero and the two accessed footprints must be disjoint. The ordinary
CUDA guard succeeds. Empty input has its separate original failure theorem. -/
specification relu_strided_buffer_real (rest : Algebra ℝ) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (positive : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : State ℝ) (xs : Nat → ℝ)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    VeriTile.Spec.Real (∃ final,
      runHost (realBackend rest) input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (realResult xs i)) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity)) :=
  real_complete rest input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
    hin hout hstride hsep s xs values

/-- This is the original exception before device guard/kernel launch; no
successful empty tensor is silently substituted for the source behavior. -/
theorem relu_strided_buffer_empty (rest : Algebra ℝ) (input output : Buffer)
    (guardSucceeds : Bool) (s : State ℝ)
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    VeriTile.Spec.Real (runHost (realBackend rest) input output guardSucceeds s =
      .hostFailure .tileDivision s) :=
  StructuralHost.empty_failure _ input output guardSucceeds s hin hout

#axiomsClean relu_strided_buffer_real
#axiomsClean relu_strided_buffer_empty
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
