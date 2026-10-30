import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelExecution
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def inputOffset (p : Parameters) (row group lane : Nat) : Nat := row*(p.groups*8)+group*8+lane
def codeOffset (p : Parameters) (row group lane : Nat) : Nat := row*(p.groups*8)+group*8+lane
def scaleOffset (p : Parameters) (row group : Nat) : Nat := row*p.groups+group

def loaded [Inhabited α] (op : Ops α) (p : Parameters) (s : State α)
    (i : TileIndex [p.block,8]) : α :=
  if i.1.val<p.groups then
    (s.mem "K" (inputOffset p (s.pids 0*p.heads+s.pids 1) i.1.val i.2.1.val)).read .real
  else op.zero

def maximum [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (g : Fin p.block) : α :=
  op.maximum (fun lane : Fin 8 => op.abs (loaded op p s (g,lane,PUnit.unit)))
def scale [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (g : Fin p.block) : α :=
  op.divide (maximum op p s g) op.levels
def divisor [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (g : Fin p.block) : α :=
  if op.equal (maximum op p s g) op.zero then op.one else scale op p s g
def code [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (i : TileIndex [p.block,8]) : Int :=
  op.convert (op.divide (loaded op p s i) (divisor op p s i.1))

def writtenCodes [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (row : Nat) : State α :=
  (TileShape.allIndices [p.block,8]).foldl (fun t i =>
    if i.1.val<p.groups then t.write "O" (codeOffset p row i.1.val i.2.1.val) (.mk .int (code op p s i)) else t) s

def written [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (row : Nat) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i =>
    if i.1.val<p.groups then t.write "S" (scaleOffset p row i.1.val) (.mk .real (scale op p s i.1)) else t)
      (writtenCodes op p s row)

theorem index_toNat (a b c d e f : Nat) :
    Int.toNat ((a : Int)*(b*c)+d*e+f) = a*(b*c)+d*e+f := by
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem index8_toNat (a b c d e f : Nat) :
    Int.toNat ((a : Int)*(b*c*8)+d*(c*8)+e*8+f) = a*(b*c*8)+d*(c*8)+e*8+f := by
  have h : (a : Int)*(b*c*8)+d*(c*8)+e*8+f =
      ((a*(b*c*8)+d*(c*8)+e*8+f : Nat) : Int) := by push_cast; ring
  rw [h,Int.toNat_natCast]

theorem execution_memory [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (dest : Nat)
    (hd : (s.mem "D" (s.pids 0)).read .int=(dest : Int)) :
    (execution op p s).val.mem=(written op p s (dest*p.heads+s.pids 1)).mem := by
  have hzero : (0.0 : ℝ)=0 := by norm_num
  have hone : (1.0 : ℝ)=1 := by norm_num
  have hlevels : (127.0 : ℝ)=127 := by norm_num
  simp [execution,written,writtenCodes,loaded,maximum,scale,divisor,code,
    inputOffset,codeOffset,scaleOffset,hd,hzero,hone,hlevels,masked_scatter_memory,
    reduction,TileShape.insertAxisIndex,TileShape.axisDim,
    index_toNat,index8_toNat,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc]

theorem codeOffset_injective (p : Parameters) (row : Nat) :
    Function.Injective (fun i : TileIndex [p.block,8] => codeOffset p row i.1.val i.2.1.val) := by
  intro i j h
  have he : i.1.val*8+i.2.1.val=j.1.val*8+j.2.1.val := by
    simp only [codeOffset] at h
    omega
  have hc : i.2.1.val=j.2.1.val := by omega
  have hg : i.1.val=j.1.val := by omega
  rcases i with ⟨i,c,u⟩
  rcases j with ⟨j,d,v⟩
  have hi : i=j := Fin.ext hg
  have hd : c=d := Fin.ext hc
  cases hi; cases hd; rfl

theorem scaleOffset_injective (p : Parameters) (row : Nat) :
    Function.Injective (fun i : TileIndex [p.block] => scaleOffset p row i.1.val) := by
  rintro ⟨i,u⟩ ⟨j,v⟩ h
  have hi : i=j := Fin.ext (Nat.add_left_cancel h)
  cases hi; rfl

theorem written_code [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [p.block,8]) (hi : i.1.val<p.groups) :
    (written op p s row).mem "O" (codeOffset p row i.1.val i.2.1.val)=.mk .int (code op p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact State.masked_scatter_readback _ _ _ _ _ i hi (codeOffset_injective p row)

theorem written_scale [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [p.block]) (hi : i.1.val<p.groups) :
    (written op p s row).mem "S" (scaleOffset p row i.1.val)=.mk .real (scale op p s i.1) :=
  State.masked_scatter_readback _ _ _ _ _ i hi (scaleOffset_injective p row)

theorem written_frame [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) (row : Nat)
    (r : RegionName) (o : Nat)
    (codes : r≠"O" ∨ ∀ g c,g<p.groups → c<8 → o≠codeOffset p row g c)
    (scales : r≠"S" ∨ ∀ g,g<p.groups → o≠scaleOffset p row g) :
    (written op p s row).mem r o=s.mem r o := by
  unfold written
  rw [State.masked_scatter_frame]
  · unfold writtenCodes
    apply State.masked_scatter_frame
    rcases codes with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ _ hi i.2.1.isLt)
  · rcases scales with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ hi)

#axiomsClean execution_memory
#axiomsClean written_code
#axiomsClean written_scale
#axiomsClean written_frame
#specNonCircular code avoiding [execution,FP.Structural.exec,kernel,written]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Kernel
