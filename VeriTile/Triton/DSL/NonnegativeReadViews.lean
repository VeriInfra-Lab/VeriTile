import VeriTile.Triton.DSL.ClosedFloatDTypes
import Mathlib.Tactic

/-! Caller-selected mathematical views of read-only signed32 metadata.
The ABI signature remains signed32. A Region.nat argument supplies its existing
nonnegative decoded view, so source indexing uses the natural carrier. This is
an interpretation of admitted inputs, not an unchecked signed-to-unsigned ABI
change: callers must establish the nonnegative/read binding on every accessed
cell. `signed32_read` is the local word-decoding law used by that connection.
No source statement, conversion, memory access or store is inserted or deleted.
These views cannot be used for output pointers or source dtype inspection. -/
namespace VeriTile.Triton.DSL.NonnegativeReadViews
open Lean Elab Term Meta

syntax (name := withNaturalReads) "with_triton_natural_reads" "[" ident,* "]" " in " term : term

private partial def inspectNames (names : List String) (raw : Syntax) : MacroM Unit := do
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes.toString
    if names.any (fun port=>name.startsWith (port++".dtype")) then
      Macro.throwErrorAt raw "a natural read view cannot inspect its storage dtype"
  for child in raw.getArgs do inspectNames names child

private def carrierProfile (profile : CallSignature) (names : List String) : CallSignature :=
  { profile with pointers := profile.pointers.map fun pair =>
      if names.contains pair.1 then (pair.1,SourceDType.u32) else pair }

@[term_elab withNaturalReads]
def elaborateViews : TermElab := fun stx expected=>do
  let `(with_triton_natural_reads [$ports:ident,*] in $invocation:term):=stx
    | throwUnsupportedSyntax
  unless invocation.raw.getKind==``CallSignature.withCall do
    throwError "natural read views require one complete with_triton_call invocation"
  let names:=ports.getElems.toList.map (·.getId.eraseMacroScopes.toString)
  unless !names.isEmpty && names.eraseDups==names do throwError "natural read ports must be nonempty and distinct"
  for port in ports.getElems do
    discard <| elabTermEnsuringType port (← mkAppM ``Region #[mkConst ``TileDType.nat])
  let original:TSyntax `term:=⟨invocation.raw[1]⟩
  let profile ← elabTermEnsuringType original (mkConst ``CallSignature)
  for name in names do
    let query ← mkAppM ``List.lookup #[mkStrLit name,← mkAppM ``CallSignature.pointers #[profile]]
    let query ← whnf query
    unless query.isAppOfArity ``Option.some 2 && (← whnf query.appArg!).isConstOf ``SourceDType.i32 do
      throwError "natural read views require signed32 pointer entries in the original ABI"
  let prepared ← PythonFullConstructors.prepareSource invocation
  let raw:=prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body=>pure body | .error message=>throwErrorAt raw "{message}"
  liftMacroM <| inspectNames names parsed
  let `(triton { $statements:tritonStmt* }):=parsed | throwError "complete source body required"
  let (_,writes):=Metadata.blockRegions statements.toList
  for write in writes do
    if write.raw.isIdent && names.contains write.raw.getId.eraseMacroScopes.toString then
      throwErrorAt stx "natural read views are read-only"
  let literals : Array (TSyntax `term) := ports.getElems.map fun port =>
    ⟨Syntax.mkStrLit port.getId.eraseMacroScopes.toString⟩
  let namesTerm ← liftMacroM `([$literals,*])
  let projected ← liftMacroM `(carrierProfile $original:term $namesTerm:term)
  elabTerm (invocation.raw.setArg 1 projected) expected

/-- On admitted nonnegative signed32 words, the natural decoded view reads
exactly the same mathematical integer; no cast or load identity is assumed. -/
theorem signed32_read (value : Int) (hi : 0 ≤ value) (bound : value < 2147483648) :
    (BitVec.ofInt 32 value).toInt = value ∧ (BitVec.ofInt 32 value).toNat = value.toNat := by
  have hword : (BitVec.ofInt 32 value).toInt = value := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  refine ⟨hword, ?_⟩
  have lo : value < 4294967296 := by omega
  simp [BitVec.toNat_ofInt, Int.emod_eq_of_lt hi lo]
end VeriTile.Triton.DSL.NonnegativeReadViews
