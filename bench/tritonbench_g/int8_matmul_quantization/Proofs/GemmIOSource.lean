import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOBinding

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000
variable {F I H : Type} [Inhabited F]

/-- Every complete original source CTA succeeds and frames all undeclared
cells. The caller's current C values remain arbitrary. -/
theorem source_result (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool)
    (p : Program s c) (xs : Inputs F s rowScale) (base : State F)
    (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    ∃ final,(invocation M carrier ops zero plans rowScale output s c columnMajor p).run base=some final ∧
      (layout F s rowScale output columnMajor).Frame xs base final := by
  obtain ⟨final,ran,_,frame⟩:=I32Safe.Gemm.program_complete M carrier ops plans atomicHalf atomicSingle
    half single rowScale output (I32Safe.Gemm.parameters s c columnMajor) (input default s rowScale xs)
    (Grid.atProgram base p.1 p.2)
    (Grid.reads_program s c columnMajor rowScale _ base (reads default s c rowScale output columnMajor xs base signed binding) _ _)
    (max_eq_left (columns_pos s c p.1 p.1.isLt)) rfl
  refine ⟨final,(invocation_run M carrier ops zero plans rowScale output s c columnMajor p base).trans ran,?_⟩
  intro r off outside
  apply frame
  rcases outside_exterior s rowScale output columnMajor xs r off outside with hr | ho
  · exact Or.inl hr
  · right
    intro i active
    have bound:=RowwiseQuantizationTriton.linear_bound s.M s.N
      (KernelGemm.rowBlock (I32Safe.Gemm.parameters s c columnMajor) (Grid.atProgram base p.1 p.2)*c.BM+i.1.val)
      (KernelGemm.columnBlock (I32Safe.Gemm.parameters s c columnMajor) (Grid.atProgram base p.1 p.2)*c.BN+i.2.1.val)
      active.1 active.2
    have stride : max s.N 1=s.N:=max_eq_left (columns_pos s c p.1 p.1.isLt)
    change off≠max s.N 1*(KernelGemm.rowBlock (I32Safe.Gemm.parameters s c columnMajor)
      (Grid.atProgram base p.1 p.2)*c.BM+i.1.val)+1*(KernelGemm.columnBlock
        (I32Safe.Gemm.parameters s c columnMajor) (Grid.atProgram base p.1 p.2)*c.BN+i.2.1.val)
    rw [stride,Nat.one_mul,Nat.mul_comm s.N]
    omega

/-- Exact selected output cell of a successful complete source execution.
It depends on other output cells only through their independently framed values. -/
theorem source_cell (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool)
    (p : Program s c) (xs : Inputs F s rowScale) (base final : State F)
    (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base)
    (ran : (invocation M carrier ops zero plans rowScale output s c columnMajor p).run base=some final)
    (cell : MatrixCell s) :
    final.mem "C" (Grid.outputOffset s cell)=
      if p.1=owner s c cell then
        Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
          (input default s rowScale xs) (cell,p.2) (base.mem "C" (Grid.outputOffset s cell))
      else base.mem "C" (Grid.outputOffset s cell) := by
  have hr:=reads default s c rowScale output columnMajor xs base signed binding
  rw [invocation_run] at ran
  split_ifs with owns
  · obtain ⟨physical,result,logical,executed,value⟩:=Grid.source_event M carrier ops plans atomicHalf atomicSingle
      half single rowScale output s c columnMajor (input default s rowScale xs) base hr (cell,p.2)
    have ownerEq:=physical_owner s c physical
    have cellEq:=congrArg (fun e : Int8MatmulQuantization.Commit s c=>e.1) logical
    change physical.logical.1=cell at cellEq
    rw [cellEq] at ownerEq
    have pidEq : physical.pid=p.1:=ownerEq.trans owns.symm
    have splitEq : physical.split=p.2:=congrArg Prod.snd logical
    rw [pidEq,splitEq,ran] at executed
    cases Option.some.inj executed
    exact value
  · obtain ⟨result,executed,_,frame⟩:=I32Safe.Gemm.program_complete M carrier ops plans atomicHalf atomicSingle
      half single rowScale output (I32Safe.Gemm.parameters s c columnMajor) (input default s rowScale xs)
      (Grid.atProgram base p.1 p.2) (Grid.reads_program s c columnMajor rowScale _ base hr _ _)
      (max_eq_left (columns_pos s c p.1 p.1.isLt)) rfl
    rw [ran] at executed
    cases Option.some.inj executed
    apply frame
    right
    intro lane active same
    let physical : PhysicalCommit s c:=⟨p.1,lane,active,p.2⟩
    have offset:=Grid.physical_offset s c columnMajor base physical
    have cellEq : physical.logical.1=cell:=Grid.output_injective s (offset.symm.trans same.symm)
    have ownerEq:=physical_owner s c physical
    rw [cellEq] at ownerEq
    exact owns ownerEq

/-- Complete source locality at every arbitrary typed current output patch. -/
theorem source_locality (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (xs : Inputs F s rowScale) (base : State F) (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    (io M carrier ops zero plans rowScale output s c columnMajor q).SourceLocality xs base := by
  intro event _ key _ left right same l r ranL ranR
  rcases key with ⟨port,j⟩
  have hp : port=0:=Fin.eq_zero port
  subst port
  obtain ⟨row,column,u⟩:=j
  cases u
  have lv:=source_cell M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
    event xs _ l signed (patch_inputs s rowScale output columnMajor xs left base binding) ranL (row,column)
  have rv:=source_cell M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
    event xs _ r signed (patch_inputs s rowScale output columnMajor xs right base binding) ranR (row,column)
  change l.mem "C" (Grid.outputOffset s (row,column))=r.mem "C" (Grid.outputOffset s (row,column))
  rw [lv,rv,patch_cell,patch_cell,same]

theorem source_coverage (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (xs : Inputs F s rowScale) (base : State F) (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    (io M carrier ops zero plans rowScale output s c columnMajor q).SourceCoverage xs base := by
  intro key _ event _ absent current final ran
  rcases key with ⟨port,j⟩
  have hp : port=0:=Fin.eq_zero port
  subst port
  obtain ⟨row,column,u⟩:=j
  cases u
  have value:=source_cell M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
    event xs _ final signed (patch_inputs s rowScale output columnMajor xs current base binding) ran (row,column)
  have other : event.1≠owner s c (row,column):=fun h=>absent ((events_mem s c (row,column) event).mpr h)
  rw [if_neg other,patch_cell] at value
  exact value

#axiomsClean source_result
#axiomsClean source_cell
#axiomsClean source_locality
#axiomsClean source_coverage
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
