import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOSource

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000
variable {F I H : Type} [Inhabited F]

def transition (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F→F→F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (xs : Inputs F s rowScale)
    (cell : MatrixCell s) (split : Fin c.S) (old : Value F (dtype output)) : Value F (dtype output) :=
  (Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
    (input default s rowScale xs) (cell,split) (FP.Structural.Cell.mk (dtype output) old)).read (dtype output)

theorem transition_typed (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F→F→F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (xs : Inputs F s rowScale)
    (cell : MatrixCell s) (split : Fin c.S) (old : Value F (dtype output)) :
    Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
      (input default s rowScale xs) (cell,split) (FP.Structural.Cell.mk (dtype output) old)=
        FP.Structural.Cell.mk (dtype output)
          (transition M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs cell split old) := by
  cases output <;> simp [transition,Grid.action]

theorem fold_typed (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F→F→F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (xs : Inputs F s rowScale)
    (cell : MatrixCell s) (splits : List (Fin c.S)) (old : Value F (dtype output)) :
    splits.foldl (fun v z=>Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
      (input default s rowScale xs) (cell,z) v) (FP.Structural.Cell.mk (dtype output) old)=
      FP.Structural.Cell.mk (dtype output)
        (splits.foldl (fun v z=>transition M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
          xs cell z v) old) := by
  induction splits generalizing old with
  | nil=>rfl
  | cons z rest ih=>rw [List.foldl_cons,transition_typed,ih];rfl

def finalValues (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F→F→F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) (xs : Inputs F s rowScale)
    (initialZero : F) : (layout F s rowScale output columnMajor).RawOutputs F :=
  fun ⟨port,j⟩=>match port with
  | ⟨0,_⟩=>(projection (j.1,j.2.1) q.events).foldl
      (fun v z=>transition M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs (j.1,j.2.1) z v)
      (value output initialZero)

/-- Every cell transition is the exact cell of a complete successful source
invocation. The order is the projection of the original arbitrary interleaving. -/
theorem trace_fold (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (xs : Inputs F s rowScale) (base : State F) (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base)
    (programs : (io M carrier ops zero plans rowScale output s c columnMajor q).Programs xs base)
    (initial : (layout F s rowScale output columnMajor).RawOutputs F) (cell : MatrixCell s)
    (splits : List (Fin c.S)) (old : Value F (dtype output)) :
    (io M carrier ops zero plans rowScale output s c columnMajor q).SourceCellTrace xs base initial
      ⟨0,(cell.1,cell.2,())⟩ (splits.map (fun z=>(owner s c cell,z))) old
      (splits.foldl (fun v z=>transition M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs cell z v) old) := by
  induction splits generalizing old with
  | nil=>
    exact IntegerDotPerAddressProgramIO.SourceCellTrace.nil
      (io:=io M carrier ops zero plans rowScale output s c columnMajor q)
      (vals:=xs) (base:=base) (initial:=initial) (key:=⟨0,(cell.1,cell.2,())⟩) old
  | cons z rest ih=>
    simp only [List.map_cons,List.foldl_cons]
    apply IntegerDotPerAddressProgramIO.SourceCellTrace.cons_of_programs
      (io M carrier ops zero plans rowScale output s c columnMajor q) xs base initial
      ⟨0,(cell.1,cell.2,())⟩ (owner s c cell,z)
      (rest.map (fun z=>(owner s c cell,z))) old
      (transition M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs cell z old)
      _ programs (program_mem s c _)
    · intro final ran
      have val:=source_cell M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
        (owner s c cell,z) xs _ final signed
        (patch_inputs s rowScale output columnMajor xs _ base binding) ran cell
      rw [if_pos rfl] at val
      rw [patch_cell] at val
      simp only [MemoryIO.updateOutput_same] at val
      exact val.trans (transition_typed M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs cell z old)
    · exact ih _

/-- The retained arbitrary-prefix theorem uses the same complete source as
this shared invocation. This preserves all cross-cell interleavings, even
prefixes that are not prefixes of the selected complete schedule. -/
theorem source_at_arbitrary_prefix (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool)
    (xs : Inputs F s rowScale) (base : State F) (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base)
    (initialZero : F) (before : List (Int8MatmulQuantization.Commit s c)) (event : Int8MatmulQuantization.Commit s c) :
    let act:=Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor (input default s rowScale xs)
    let current:=Grid.run s c act before ((layout F s rowScale output columnMajor).patchOutputs xs (initialValues initialZero s output) base)
    ∃ final,(invocation M carrier ops zero plans rowScale output s c columnMajor
      (owner s c event.1,event.2)).run current=some final ∧
      final.mem "C" (Grid.outputOffset s event.1)=
        (Grid.advance s c act current event).mem "C" (Grid.outputOffset s event.1) := by
  dsimp only
  rw [patch_reset]
  obtain ⟨physical,final,logical,ran,cell⟩:=Grid.source_at_prefix M carrier ops plans atomicHalf atomicSingle
    half single rowScale output s c columnMajor (input default s rowScale xs) base
    (reads default s c rowScale output columnMajor xs base signed binding) (Grid.zeroCell output initialZero) before event
  have pid:=physical_owner s c physical
  have first : physical.logical.1=event.1:=congrArg Prod.fst logical
  have second : physical.split=event.2:=congrArg Prod.snd logical
  rw [first] at pid
  rw [pid,second] at ran
  exact ⟨final,(invocation_run M carrier ops zero plans rowScale output s c columnMajor _ _).trans ran,cell⟩

#axiomsClean fold_typed
#axiomsClean trace_fold
#axiomsClean source_at_arbitrary_prefix
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
