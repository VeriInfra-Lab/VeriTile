import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealSequence
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealStore

/-! Exact forward output addressing and full-chunk store mask. Live padding
in a shortened final chunk is written, rather than left untouched. The shared
store fold supplies tagged readback and preservation of every other cell. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStore
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def basePointer (region : RegionName) (batch chunk head : Nat) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase region)
    (.castIntToNat (.add .int .nil
      (.add .int .nil (.mul .int .nil (.ref .int [] "pid_b") (.castNatToInt (.constNat batch)))
        (.mul .int .nil (.ref .int [] "pid_c") (.castNatToInt (.constNat chunk))))
      (.mul .int .nil (.ref .int [] "pid_h") (.castNatToInt (.constNat head)))))

def pointers (BM BN row column : Nat) : Op .ptr [BM, BN] :=
  .ptrAdd .scalarL (.ref .ptr [] "out_ptr")
    (.castIntToNat (.add .int Broadcast.nil.consL.consR
      (.mul .int .scalarL (.castNatToInt (.constNat row))
        (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "offs_m")))
      (.mul .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "offs_n"))
        (.castNatToInt (.constNat column)))))

def mask (BM BN chunk : Nat) : Op .bool [BM, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .int .scalarR (.expandDim ⟨1, by simp⟩ (.ref .int [BM] "offs_m")) (.castNatToInt (.constNat chunk)))
    (.lt .int .scalarR (.expandDim ⟨0, by simp⟩ (.ref .int [BN] "offs_n")) (.castNatToInt (.constNat chunk)))

def code (BM BN chunk : Nat) : Stmt :=
  .store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
    (.ref .real [BM, BN] "out") (.mask (mask BM BN chunk))

abbrev pointerValues := BmmChunkBwd.RealResidual.pointerValues
abbrev offset := BmmChunkBwd.RealStore.offset

def active (BM BN chunk : Nat) (rows : Fin BM → Nat)
    (columns : Fin BN → Nat) (i : TileIndex [BM, BN]) : Prop :=
  rows i.1 < chunk ∧ columns i.2.1 < chunk
instance (BM BN chunk : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat) :
    DecidablePred (active BM BN chunk rows columns) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def write (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ) : BlockState :=
  BmmChunkBwd.RealStore.write s region BM BN base row column chunk chunk rows columns values

theorem base_eval (s : BlockState) (region : RegionName) (batch chunk head pb pc ph : Nat)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int))) :
    evalOp (basePointer region batch chunk head) s =
      some (Tile.scalar (region, pb * batch + pc * chunk + ph * head)) := by
  simp [basePointer, evalOp.eq_def, hb, hc, hh, bind, Option.bind]
  apply Tile.ext
  intro i
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Nat.zero_add]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]

theorem pointers_eval (s : BlockState) (region : RegionName) (BM BN base row column : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [] "out_ptr" = some (Tile.scalar (region, base)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (rows i : Int))))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (columns i : Int)))) :
    evalOp (pointers BM BN row column) s =
      some (pointerValues region BM BN base row column rows columns) := by
  simp [pointers, evalOp.eq_def, hp, hr, hn, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Tile.mat, pointerValues,
    BmmChunkBwd.RealResidual.pointerValues, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Tile.expandDim, TileShape.dropInsertedIndex, Nat.mul_comm, -Tile.scalar_eta]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  ac_rfl

theorem mask_eval (s : BlockState) (BM BN chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (rows i : Int))))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (columns i : Int)))) :
    evalOp (mask BM BN chunk) s =
      some (Tile.mat fun i j => decide (rows i < chunk ∧ columns j < chunk)) := by
  simp [mask, evalOp.eq_def, hr, hn, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
  simp [Int.ofNat_lt]

theorem executes (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hp : s.regs .ptr [BM, BN] "out_ptrs" = some (pointerValues region BM BN base row column rows columns))
    (hv : s.regs .real [BM, BN] "out" = some (Tile.mat fun i j => some (values i j)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec (fun i => (rows i : Int))))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => (columns i : Int)))) :
    stepStmt (code BM BN chunk) s =
      some (write s region BM BN base row column chunk rows columns values) := by
  have hm := mask_eval s BM BN chunk rows columns hr hn
  rw [code, stepStmt]
  simp only [evalOp_ref, hp, hv, hm, bind, Option.bind_some,
    pointerValues, BmmChunkBwd.RealResidual.pointerValues, Tile.mat,
    Option.map_some, decide_eq_true_eq, BlockState.writeMemTyped_real]
  rfl

theorem frame (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (r : RegionName) (address : Nat)
    (h : ∀ i, active BM BN chunk rows columns i →
      ¬ (region = r ∧ offset BM BN base row column rows columns i = address)) :
    (write s region BM BN base row column chunk rows columns values).mem r address = s.mem r address :=
  BmmChunkBwd.RealStore.frame s region BM BN base row column chunk chunk rows columns values r address h

theorem readback (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (i : TileIndex [BM, BN]) (hi : active BM BN chunk rows columns i)
    (hinj : ∀ a b, active BM BN chunk rows columns a → active BM BN chunk rows columns b →
      offset BM BN base row column rows columns a = offset BM BN base row column rows columns b → a = b) :
    (write s region BM BN base row column chunk rows columns values).mem region
      (offset BM BN base row column rows columns i) = MemCell.real (values i.1 i.2.1) :=
  BmmChunkBwd.RealStore.readback s region BM BN base row column chunk chunk rows columns values i hi hinj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStore
