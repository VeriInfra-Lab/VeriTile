import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeCompilation

/-! Native FP32 composition helper. The compiler's local SOURCE-SITE graphs
are bound to this candidate hash, template, configuration and FP32 runtime
scale ABI. The complete source interpreter is then proved equivalent to the
independent native Word recurrences. The selected graphs preserve contraction,
operand order, zero-carry specialization and every original FP32 memory bit.

Local PTX instruction interpretations and local compiler selection remain
explicit boundaries. No split=fused law, whole-output oracle, finite/normal
input restriction or native=Real assertion occurs. All raw scale/input words
are admitted. The inherited Real theorem remains independent.

The result includes None, original destination identities, ordered last-write
bytes for arbitrary aliases, caller byte frame, and all existing failures.
Preparation failure requires no placement, allocation or compilation witness.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.NativeFPEquiv
open VeriTile.Triton FP.Structural PackedStorage Native
noncomputable section
set_option maxHeartbeats 5000000

abbrev Byte := BitVec 8
def negativeWord (negative : Bool) (value : Word) : Word :=
  if negative then negBits value else value
def encodeWord (negative : Bool) (value : Word) (byte : Fin 4) : Byte :=
  BitVec.ofNat 8 ((negativeWord negative value).toNat / 2^(8*byte.val))
def decodeWord (negative : Bool) (bytes : Fin 4 → Byte) : Word :=
  negativeWord negative (word ((bytes 0).toNat + 256*(bytes 1).toNat +
    65536*(bytes 2).toNat + 16777216*(bytes 3).toNat))
def raw (value : Cell Node) : Word := (value.read .real).value
def decode (negative : Bool) (bytes : Fin 4 → Byte) : Cell Node :=
  .mk .real (atom (decodeWord negative bytes))
def encode (negative : Bool) (value : Cell Node) : Fin 4 → Byte :=
  encodeWord negative (raw value)

def reference (p : Instructions) (h : Call) (kind : Kind) (scale : Word)
    (targets : RegionName → Host.Target) (before : Copyback.Heap Byte) :=
  fun r i => referenceWords p h kind scale (Host.logicalData h targets decodeWord before)
    r (Logical.coordinates h i)

def run (compiler : Compiler) (h : Call) (kind : Kind) (scale : Word)
    (api : WholeHost.API h kind Byte (Cell Node)) (targets : RegionName → Host.Target)
    (before : Copyback.Heap Byte) := compiledRun compiler h kind scale api targets encode before

/-- No kernel result is a premise. Source success, all iterations, raw FP32
values and the complete ordered caller-byte result are derived. -/
theorem whole_wrapper_equiv (compiler : Compiler) (p : Instructions)
    (h : Call) (kind : Kind) (scale : Word)
    (api : WholeHost.API h kind Byte (Cell Node)) (targets : RegionName → Host.Target)
    (before : Copyback.Heap Byte) (live : Nat → Prop)
    (ordinaryPreparation : WholeHost.PreparationLaws h kind api targets decode before live)
    (ordinaryCopy : WholeHost.CopyLaws api)
    (localCompilation : LocalCompilation compiler p h kind api) :
    Host.Outcome h kind targets encodeWord (reference p h kind scale targets before) before
      (run compiler h kind scale api targets before) := by
  unfold run
  rw [compiled_run_eq compiler p h kind scale api targets encode before localCompilation]
  have source := FPEquiv.whole_wrapper_equiv (algebra p selected kind scale) h kind scaleSlot
    api targets decode encode before live ordinaryPreparation ordinaryCopy
  have refs : (fun r i => FPLogical.reference (algebra p selected kind scale) h kind scaleSlot
      (Host.logicalData h targets decode before) r (Logical.coordinates h i))=
      (fun r i => outputCell kind (Logical.chunkSpan h (Logical.coordinates h i).2.1.val) r
        ((Logical.coordinates h i).2.1.val-Logical.chunkStart h (Logical.coordinates h i).2.1.val)
        (reference p h kind scale targets before r i)) := by
    funext r i
    rw [reference_lowering]
    rfl
  rw [refs] at source
  have jobs : Host.allJobs h kind targets encode
      (fun r i => outputCell kind (Logical.chunkSpan h (Logical.coordinates h i).2.1.val) r
        ((Logical.coordinates h i).2.1.val-Logical.chunkStart h (Logical.coordinates h i).2.1.val)
        (reference p h kind scale targets before r i)) =
      Host.allJobs h kind targets encodeWord (reference p h kind scale targets before) := by
    unfold Host.allJobs
    apply congrArg (fun f => (Host.order kind).flatMap f)
    funext r
    simp only [Host.roleJobs,Copyback.jobs,encode,raw,outputCell_value]
  simpa only [Host.Outcome,jobs] using source

/-- No successful compiler/setup witness on preparation error, even when the
dimensions cannot fit any allocation. Empty calls return before preparation. -/
theorem preparation_failure (compiler : Compiler) (h : Call) (kind : Kind) (scale : Word)
    (api : WholeHost.API h kind Byte (Cell Node)) (targets : RegionName → Host.Target)
    (before : Copyback.Heap Byte) (message : String) (failed : api.prepare=.error message) :
    run compiler h kind scale api targets before=
      if Logical.cells h=0 then .ok ⟨(),targets,before⟩ else .error ⟨.preparation,message,before⟩ :=
  WholeHost.preparation_failure h kind api targets encode _ _ before message failed

abbrev validation_failure := @Host.validation_failure
abbrev negative_dimension := Host.negative_dimension
abbrev invalid_tile := Host.invalid_tile
abbrev exact_last_write := @Copyback.exact_last_write
abbrev no_overlap_readback := @Copyback.view_readback
abbrev physical_addresses := NativeABI.source_addresses

#axiomsClean whole_wrapper_equiv
#axiomsClean preparation_failure
#specNonCircular reference avoiding [run,FPStorage.execute,FPLogical.reference]
#print axioms whole_wrapper_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.NativeFPEquiv
