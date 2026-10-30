import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Binding
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Certified
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Native
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.L2NormBwd
open VeriTile.Triton Physical Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section

def nativeResult {M : Type} (ops : Ops M Native.Word Words.Byte) (target : Native.Instructions)
    (call : Call) (eps : Native.Word) (initial : World M) (i : Nat) : Native.Word :=
  Native.row target call.cols (blockSize call.cols) eps
    (fun j => logical ops Words.signNegation initial.memory call.x ((i/call.cols)*call.cols+j))
    (fun j => logical ops Words.signNegation initial.memory call.dy ((i/call.cols)*call.cols+j)) (i%call.cols)

/-- Unconditional raw native instruction-graph result. Every primitive is
opaque over all 2^32 words; there are no numerical correspondence premises. -/
theorem source_native {M : Type} (ops : Ops M Native.Word Words.Byte) (rt : Runtime M)
    (lib : Library ops Words.signNegation rt) (grids : Machine.GridContract rt) (actual : Native.Instructions)
    (call : Call) (eps : Native.Word) (initial : World M) (domain : InputDomain call initial) :
    Ensures (fun output after => WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after ∧
      Nonempty (NativeCertificate ops actual rt call eps initial output after) ∧
      ∀ i,i<call.x.numel → logical ops Words.signNegation after.memory output i=nativeResult ops actual call eps initial i)
      (fun _ after => Frame ops call.caller initial after)
      (Host.run ops (Native.reduction actual) rt call eps initial) := by
  have spec := whole_spec ops Words.signNegation (Native.reduction actual) rt lib call eps initial domain
  cases outcome : Host.run ops (Native.reduction actual) rt call eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    change WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after at spec
    exact ⟨spec,native_certificate ops Words.signNegation actual rt lib grids call eps initial after output domain outcome,spec.values⟩

/-- The optional primitive model bridge consists of six local primitive
function equalities. Exact returned words and all four actual backing bytes
follow from the whole program theorem, not from output/address assumptions. -/
theorem source_fp {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (rt : Runtime M) (lib : Library ops Words.signNegation rt) (grids : Machine.GridContract rt) (actual target : Native.Instructions)
    (numeric : Native.Obligations actual target) (call : Call) (eps : Native.Word)
    (initial : World M) (domain : InputDomain call initial) :
    Ensures (fun output after => WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after ∧
      Nonempty (NativeCertificate ops actual rt call eps initial output after) ∧
      ∀ i,i<call.x.numel → logical ops Words.signNegation after.memory output i=nativeResult ops target call eps initial i ∧
        ∀ byte,byte<4 → ops.observe after.memory (output.key i).owner ((output.key i).address+byte)=
          (Words.signNegation.encode output.negative (nativeResult ops target call eps initial i)).extractLsb' (8*byte) 8)
      (fun _ after => Frame ops call.caller initial after)
      (Host.run ops (Native.reduction actual) rt call eps initial) := by
  have spec := source_native ops rt lib grids actual call eps initial domain
  cases outcome : Host.run ops (Native.reduction actual) rt call eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    change WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after ∧ _ at spec
    refine ⟨spec.1,spec.2.1,?_⟩
    intro i hi
    have exactWord : logical ops Words.signNegation after.memory output i=nativeResult ops target call eps initial i := by
      rw [spec.2.2 i hi,Native.instructions_eq actual target numeric]
    exact ⟨exactWord,fun byte small => Words.logical_bytes ops codec after.memory output i _ exactWord byte small⟩

/-- Checked native graph theorem with exact direct-view/returned-storage
identity and the historical source, integer and launch certificates. -/
theorem whole_native {M : Type} (ops : Ops M Native.Word Words.Byte) (rt : Runtime M)
    (lib : Library ops Words.signNegation rt) (grids : Machine.GridContract rt) (actual : Native.Instructions)
    (x dy : TensorMetadata) (input gradient : View) (eps : Native.Word) (initial : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : initial.live input.storage) (ld : initial.live gradient.storage) :
    let call : Call := ⟨input,gradient⟩
    match checkedRun ops (Native.reduction actual) rt x dy input gradient eps initial with
    | .error _ => True
    | .ok result => Ensures (fun output after =>
        WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after ∧
        Nonempty (NativeCertificate ops actual rt call eps initial output after) ∧
        Nonempty (DirectPath ops (Native.reduction actual) rt call eps initial output after) ∧
        ∀ i,i< input.numel → logical ops Words.signNegation after.memory output i=nativeResult ops actual call eps initial i)
        (fun _ after => Frame ops call.caller initial after) result := by
  dsimp only
  rw [checked_run _ _ _ _ _ _ _ _ _ h]
  have domain:=checked_domain x dy input gradient initial bx bd lx ld
  have spec:=source_native ops rt lib grids actual ⟨input,gradient⟩ eps initial domain
  cases outcome : Host.run ops (Native.reduction actual) rt ⟨input,gradient⟩ eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    exact ⟨spec.1,spec.2.1,direct_path ops Words.signNegation (Native.reduction actual) rt lib
      ⟨input,gradient⟩ eps initial after output domain outcome,spec.2.2⟩

/-- Exactly the six local native primitive model equalities, with complete
returned-word and four-byte readback. No arithmetic domain narrowing is added. -/
theorem whole_fp {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (rt : Runtime M) (lib : Library ops Words.signNegation rt) (grids : Machine.GridContract rt)
    (actual target : Native.Instructions) (numeric : Native.Obligations actual target)
    (x dy : TensorMetadata) (input gradient : View) (eps : Native.Word) (initial : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : initial.live input.storage) (ld : initial.live gradient.storage) :
    let call : Call := ⟨input,gradient⟩
    match checkedRun ops (Native.reduction actual) rt x dy input gradient eps initial with
    | .error _ => True
    | .ok result => Ensures (fun output after =>
        WholeSpec ops Words.signNegation (Native.reduction actual) rt call eps initial output after ∧
        Nonempty (NativeCertificate ops actual rt call eps initial output after) ∧
        Nonempty (DirectPath ops (Native.reduction actual) rt call eps initial output after) ∧
        ∀ i,i< input.numel → logical ops Words.signNegation after.memory output i=nativeResult ops target call eps initial i ∧
          ∀ byte,byte<4 → ops.observe after.memory (output.key i).owner ((output.key i).address+byte)=
            (Words.signNegation.encode output.negative (nativeResult ops target call eps initial i)).extractLsb' (8*byte) 8)
        (fun _ after => Frame ops call.caller initial after) result := by
  dsimp only
  rw [checked_run _ _ _ _ _ _ _ _ _ h]
  have domain:=checked_domain x dy input gradient initial bx bd lx ld
  have spec:=source_fp ops codec rt lib grids actual target numeric ⟨input,gradient⟩ eps initial domain
  cases outcome : Host.run ops (Native.reduction actual) rt ⟨input,gradient⟩ eps initial with
  | error reason after => rw [outcome] at spec;exact spec
  | ok output after =>
    rw [outcome] at spec
    exact ⟨spec.1,spec.2.1,direct_path ops Words.signNegation (Native.reduction actual) rt lib
      ⟨input,gradient⟩ eps initial after output domain outcome,spec.2.2⟩

#axiomsClean whole_native
#axiomsClean whole_fp

#specNonCircular nativeResult avoiding [Host.run,Tile.scalar,Native.algebra,source]
#axiomsClean source_native
#axiomsClean source_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd
