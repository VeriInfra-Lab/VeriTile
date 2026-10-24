import bench.repairs.tritonbench_g.adam_update_triton.Proofs.NativeWords

/-! Complete source-to-independent-native-graph relation. The native recurrence
retains the observed FMA contractions, operand order, unordered !=, ordered >,
and boolean-to-FP32 multiplication. Primitive instruction ports are model
boundaries, not optimization laws or per-task numerical experiments. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.NativeExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
noncomputable section

/-- Preserve the type of every observed memory cell. Only floating graph
values are interpreted; integer, predicate and pointer payloads are retained. -/
def observeValue (native : Native.Instructions α) : (d : TileDType) → Value (Term α) d → Value α d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Native.interpret native
  | .nat | .int | .bool | .ptr | .blockPtr => id

def observeCell (native : Native.Instructions α) : Cell (Term α) → Cell α
  | .mk d v => .mk d (observeValue native d v)

/-- Both native output words and the full raw regional exterior frame follow from the
successful execution of the complete source. Entry loads are word leaves,
preventing contraction across memory. All FP32 bit patterns are admitted. -/
theorem kernel_execution (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B pid : Nat) (s : State (Term Native.Word))
    (ps gs ms : Nat → Native.Word)
    (hp : ∀ j : TileIndex [B], pid * B + j.1.val < n →
      (s.mem "P" (pid * B + j.1.val)).read .real = .input (ps (pid * B + j.1.val)))
    (hg : ∀ j : TileIndex [B], pid * B + j.1.val < n →
      (s.mem "G" (pid * B + j.1.val)).read .real = .input (gs (pid * B + j.1.val)))
    (hm : ∀ j : TileIndex [B], pid * B + j.1.val < n →
      (s.mem "M" (pid * B + j.1.val)).read .real = .input (ms (pid * B + j.1.val))) :
    ∃ final,
      run (Native.semantics (NativeWords.ports native)).backend (source total n B).surfaceBody
        (Tile.programState (Native.arguments params) pid s) = some final ∧
      (∀ output (j : TileIndex [B]), pid * B + j.1.val < n →
        observeCell (NativeWords.ports native) (final.mem (Tile.outputRegion output) (pid * B + j.1.val)) =
          Cell.mk .real (Native.reference (NativeWords.ports native) params output
            (ps (pid * B + j.1.val)) (gs (pid * B + j.1.val)) (ms (pid * B + j.1.val)))) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o) := by
  refine ⟨(Tile.sourceRun (Native.semantics (NativeWords.ports native)) (Native.arguments params) total n B pid s).val,
    (Tile.sourceRun (Native.semantics (NativeWords.ports native)) (Native.arguments params) total n B pid s).property,
    ?_, Tile.frame (Native.semantics (NativeWords.ports native)) (Native.arguments params) total n B pid s⟩
  intro output j active
  rw [Tile.readback _ _ _ _ _ _ _ _ _ active, hp j active, hg j active, hm j active]
  exact congrArg (Cell.mk .real) (Native.lowering (NativeWords.ports native) params output _ _ _)

/-- Input-window form used by the shared IO interface. A word enters the graph
as a leaf, and output observations retain the source's cell dtype. -/
theorem kernel_inputs (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B pid : Nat) (s : State (Term Native.Word))
    (ps gs ms : Fin B → Native.Word)
    (hp : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "P" (pid * B + j.val)).read .real = .input (ps j))
    (hg : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "G" (pid * B + j.val)).read .real = .input (gs j))
    (hm : ∀ j : Fin B, pid * B + j.val < n →
      (s.mem "M" (pid * B + j.val)).read .real = .input (ms j)) :
    ∃ final,
      run (Native.semantics (NativeWords.ports native)).backend (source total n B).surfaceBody
        (Tile.programState (Native.arguments params) pid s) = some final ∧
      (∀ output (j : Fin B), pid * B + j.val < n →
        observeCell (NativeWords.ports native) (final.mem (Tile.outputRegion output) (pid * B + j.val)) =
          Cell.mk .real (Native.reference (NativeWords.ports native) params output (ps j) (gs j) (ms j))) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o) := by
  refine ⟨(Tile.sourceRun (Native.semantics (NativeWords.ports native)) (Native.arguments params)
    total n B pid s).val,
    (Tile.sourceRun (Native.semantics (NativeWords.ports native)) (Native.arguments params)
      total n B pid s).property, ?_,
    Tile.frame (Native.semantics (NativeWords.ports native)) (Native.arguments params)
      total n B pid s⟩
  intro output j active
  rw [Tile.readback _ _ _ _ _ _ _ output (j, PUnit.unit) active,
    hp j active, hg j active, hm j active]
  exact congrArg (Cell.mk .real) (Native.lowering (NativeWords.ports native) params output _ _ _)

#specNonCircular Native.reference avoiding [Native.interpret, Native.sourceScalar, Tile.scalar, Tile.sourceRun]
#axiomsClean kernel_execution
#axiomsClean kernel_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.NativeExecution
