import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeForwardLoop
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeGraphMemory
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.ForwardGridStores
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

def tensorAt (c : Shape) (μ : RegionName → Nat → Graph) (r : RegionName) (h row d : Nat) : Graph :=
  μ r ((h*c.rows+row)*c.dim+d)
def input (c : Shape) (μ : RegionName → Nat → Graph) : Target.Inputs :=
  ⟨tensorAt c μ "Q",tensorAt c μ "K",tensorAt c μ "V",tensorAt c μ "DO"⟩
def Inputs (mem : Grid.Memory Graph) (μ : RegionName → Nat → Graph) : Prop :=
  ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,mem r o=.mk .fp32 (μ r o)

theorem entry_readyQ (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (μ : RegionName → Nat → Graph) (hi : Inputs mem μ) :
    ReadyQ c scale (Grid.entry (Grid.forwardGrid c) pid (embed mem)) (input c μ) pid.2 := by
  intro i
  unfold forwardQ forwardOffset
  rw [Grid.forwardHead_eq]
  change (embed mem "Q" ((pid.2*c.rows+forwardQueryRow (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid (embed mem)) i.1)*c.dim+i.2.1.val)).read .fp32=_
  simp only [embed,hi "Q" (by simp)]
  rfl

theorem entry_readyKV (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (μ : RegionName → Nat → Graph) (hi : Inputs mem μ) (hp : Grid.Legal (Grid.forwardGrid c) pid)
    (block : Nat) (hb : block<c.blocks) :
    ReadyKV c scale (Grid.entry (Grid.forwardGrid c) pid (embed mem)) (input c μ) pid.2 block := by
  have aux (r : RegionName) (hr : r∈(["Q","K","V","DO"] : List RegionName)) (i : FKey (forwardParameters c scale)) :
      forwardKV (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid (embed mem)) r block i=
        fun _=>Stage.ready (tensorAt c μ r pid.2 (block*128+i.1.val) i.2.1.val) := by
    have bound:=Grid.row_bound c pid.2 block i.1.val hp.2 hb i.1.isLt
    unfold forwardKV forwardKVPointer
    rw [Grid.forwardHead_eq]
    simp only [TileShape.blockPtr_inBounds_2d_offsets_index,Nat.zero_add]
    rw [if_pos (by simp [forwardParameters,bound,i.2.1.isLt])]
    unfold forwardKVOffset
    rw [Grid.forwardHead_eq]
    change (embed mem r ((pid.2*c.rows+block*128+i.1.val)*c.dim+i.2.1.val)).read .fp32=_
    simp only [embed,hi r hr]
    simp only [tensorAt,Nat.add_assoc]
    rfl
  exact ⟨aux "K" (by simp),aux "V" (by simp)⟩

theorem forwardStep_readonly (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (r : RegionName) (o : Nat) (hr : r≠"Out" ∧ r≠"L" ∧ r≠"M") :
    store one .forward c scale owners pid mem r o=mem r o := by
  have hr' : r≠(forwardParameters c scale).output ∧ r≠(forwardParameters c scale).denominator ∧ r≠(forwardParameters c scale).maximum := hr
  change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) r o)=_
  unfold Grid.forwardStep forwardWrite
  dsimp only []
  rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inl hr'.1)]
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2.2)]
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2.1)]
  exact congrFun (congrFun (finish_embed one .forward c owners mem) r) o

theorem forwardStep_inputs (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (μ : RegionName → Nat → Graph) (hi : Inputs mem μ) :
    Inputs (store one .forward c scale owners pid mem) μ := by
  intro r hr o
  rw [forwardStep_readonly one c scale owners pid mem r o]
  · exact hi r hr o
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl <;> decide

theorem storeOwner_out (c : Shape) (owners : TritonAttention.Whole001.Native.Owners) (i : Coord c) :
    storeOwner c owners "Out" (offset c i)=(Reduction.owner i.2.2.val).val := by
  simp [storeOwner,offset,Nat.add_mod,Nat.mod_eq_of_lt i.2.2.isLt]

theorem storeOwner_l (c : Shape) (owners : TritonAttention.Whole001.Native.Owners) (h : Fin c.groups) (i : Fin c.rows) :
    storeOwner c owners "L" (rowOffset c h i)=(owners.l h.val i.val).val := by
  have hdiv : (h.val*c.rows+i.val)/c.rows=h.val := by
    rw [Nat.mul_comm h.val c.rows,Nat.mul_add_div c.rows_pos,Nat.div_eq_of_lt i.isLt,Nat.add_zero]
  simp [storeOwner,rowOffset,hdiv,Nat.mod_eq_of_lt i.isLt]

theorem storeOwner_m (c : Shape) (owners : TritonAttention.Whole001.Native.Owners) (h : Fin c.groups) (i : Fin c.rows) :
    storeOwner c owners "M" (rowOffset c h i)=(owners.m h.val i.val).val := by
  have hdiv : (h.val*c.rows+i.val)/c.rows=h.val := by
    rw [Nat.mul_comm h.val c.rows,Nat.mul_add_div c.rows_pos,Nat.div_eq_of_lt i.isLt,Nat.add_zero]
  simp [storeOwner,rowOffset,hdiv,Nat.mod_eq_of_lt i.isLt]

theorem offset_injective (p : ForwardParameters) (s : State Expr) : Function.Injective (forwardOffset p s) := by
  intro a b he
  have ha:=a.2.1.isLt
  have hb:=b.2.1.isLt
  have hcol : a.2.1=b.2.1 := by
    apply Fin.ext
    have hh:=congrArg (· % p.head) he
    simpa only [forwardOffset,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] using hh
  have hdim : 0<p.head := by omega
  have hrow : a.1=b.1 := by
    apply Fin.ext
    simp only [forwardOffset,forwardQueryRow] at he
    have hc:=congrArg Fin.val hcol
    nlinarith
  exact Prod.ext hrow (Prod.ext hcol (Subsingleton.elim _ _))

theorem forwardStep_values (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (mem : Grid.Memory Graph)
    (μ : RegionName → Nat → Graph) (hi : Inputs mem μ) (i : Coord c) :
    store one .forward c scale owners (Grid.forwardProgram c i) mem "Out" (offset c i)=
      .mk .fp32 ((Target.forward c one scale (input c μ) owners).output i.1.val i.2.1.val i.2.2.val) ∧
    store one .forward c scale owners (Grid.forwardProgram c i) mem "L" (rowOffset c i.1 i.2.1)=
      .mk .fp32 ((Target.forward c one scale (input c μ) owners).denominator i.1.val i.2.1.val) ∧
    store one .forward c scale owners (Grid.forwardProgram c i) mem "M" (rowOffset c i.1 i.2.1)=
      .mk .fp32 ((Target.forward c one scale (input c μ) owners).maximum i.1.val i.2.1.val) := by
  let pid:=Grid.forwardProgram c i
  let s:=Grid.entry (Grid.forwardGrid c) pid (embed mem)
  let lane:=Grid.forwardLane c scale i
  have hp:=Grid.forwardProgram_legal c i
  have hq:=entry_readyQ c scale pid mem μ hi
  have hkv:=entry_readyKV c scale pid mem μ hi hp
  have blocks : forwardBlocks (forwardParameters c scale) s=pid.1+1 := by
    simp [forwardBlocks,forwardStop,forwardParameters,s,Grid.entry,Grid.ids]
  have hn : pid.1+1≤c.blocks := by have:=hp.1;change pid.1<c.blocks at this;omega
  have hlo (col : Fin c.dim) : AccAt c scale one
      (forwardRounds (modelScalar one .forward) (forwardParameters c scale) s (forwardBlocks (forwardParameters c scale) s))
      lane.1 col (Target.rounds c one scale (input c μ) pid.2 (forwardQueryRow (forwardParameters c scale) s lane.1) col.val (pid.1+1)) := by
    rw [blocks]
    exact forwardRounds_native c scale one s (input c μ) pid.2 hq lane.1 col (pid.1+1)
      (by intro b hb;exact hkv b (by omega))
  obtain ⟨final,he,hm,ho,hl,hmax,_⟩:=forwardCorrectness (algebraScalar one .forward) (modelScalar one .forward)
    (forwardParameters c scale) s (by change 0<128;decide)
    (by dsimp [forwardStop,forwardParameters];exact dvd_mul_left 128 _)
    (by simp [forwardParameters]) (by simp [forwardParameters]) (by simp [forwardParameters]) (offset_injective _ _)
  have hin : (forwardOutPointer (forwardParameters c scale) s).inBounds (TileShape.indexToList _ lane) [0,1]=Bool.true := by
    unfold forwardOutPointer forwardQPointer
    dsimp only [s]
    rw [Grid.forwardHead_eq]
    simp only [TileShape.blockPtr_inBounds_2d_offsets_index,Nat.zero_add]
    apply decide_eq_true
    constructor
    · simpa [forwardParameters,Grid.entry,Grid.ids,Nat.add_assoc] using
        Grid.row_bound c pid.2 pid.1 lane.1.val hp.2 hp.1 lane.1.isLt
    · exact lane.2.1.isLt
  have hout:=ho lane hin
  have hlv:=hl (lane.1,())
  have hmv:=hmax (lane.1,())
  rw [hm] at hout hlv hmv
  change Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "Out" (forwardOffset (forwardParameters c scale) s lane)=_ at hout
  change Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "L" (forwardRowOffset (forwardParameters c scale) s (lane.1,()))=_ at hlv
  change Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "M" (forwardRowOffset (forwardParameters c scale) s (lane.1,()))=_ at hmv
  dsimp only [s,pid,lane] at hout hlv hmv
  rw [Grid.forward_target_offset] at hout
  rw [Grid.forward_target_row] at hlv hmv
  constructor
  · change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "Out" (offset c i))=_
    rw [hout]
    change Cell.mk .fp32 (Target.half (lowerScalar one .forward
      ((forwardRounds (modelScalar one .forward) (forwardParameters c scale) s (forwardBlocks (forwardParameters c scale) s)).numerator (lane.1,i.2.2,())
        (storeOwner c owners "Out" (offset c i)))))=_
    rw [storeOwner_out,(hlo i.2.2).2.2.1,replica_val]
    simp only [Target.forward]
    congr 3 <;> simp [s,pid,lane,Grid.entry,Grid.ids,Grid.forwardProgram,Grid.forwardLane,forwardQueryRow,forwardParameters,Nat.div_add_mod']
  constructor
  · change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "L" (rowOffset c i.1 i.2.1))=_
    rw [hlv]
    change Cell.mk .fp32 (lowerScalar one .forward
      ((forwardRounds (modelScalar one .forward) (forwardParameters c scale) s (forwardBlocks (forwardParameters c scale) s)).denominator (lane.1,())
        (storeOwner c owners "L" (rowOffset c i.1 i.2.1))))=_
    rw [storeOwner_l,(hlo ⟨0,c.dim_pos⟩).2.1,replica_val]
    simp only [Target.forward]
    congr 2 <;> simp [s,pid,lane,Grid.entry,Grid.ids,Grid.forwardProgram,Grid.forwardLane,forwardQueryRow,forwardParameters,Nat.div_add_mod']
  · change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "M" (rowOffset c i.1 i.2.1))=_
    rw [hmv]
    change Cell.mk .fp32 (lowerScalar one .forward
      ((forwardRounds (modelScalar one .forward) (forwardParameters c scale) s (forwardBlocks (forwardParameters c scale) s)).maximum (lane.1,())
        (storeOwner c owners "M" (rowOffset c i.1 i.2.1))))=_
    rw [storeOwner_m,(hlo ⟨0,c.dim_pos⟩).1,replica_val]
    simp only [Target.forward]
    congr 2 <;> simp [s,pid,lane,Grid.entry,Grid.ids,Grid.forwardProgram,Grid.forwardLane,forwardQueryRow,forwardParameters,Nat.div_add_mod']

#axiomsClean forwardStep_values
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
