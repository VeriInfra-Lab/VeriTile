import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealSeed
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

abbrev integer (n : Nat) : Op .int [] := .castNatToInt (.constNat n)
abbrev reg (n : RegName) : Op .int [] := .ref .int [] n
abbrev add (a b : Op .int []) := Op.add .int .nil a b
abbrev mul (a b : Op .int []) := Op.mul .int .nil a b
abbrev natReg (n : RegName) : Op .nat [] := .castIntToNat (reg n)

def keyPtr (p : Parameters) (r : RegionName) : Op .blockPtr [p.BK,p.BT] :=
  .makeBlockPtrDynMetadata r (.castIntToNat
    (add (add (mul (reg "i_bh") (reg "s_k_h"))
      (mul (mul (reg "i_k") (integer p.BK)) (reg "s_k_d")))
      (mul (mul (reg "i_t") (integer p.BT)) (reg "s_k_t"))))
    [p.BK,p.BT] [p.BK,p.BT] [natReg "s_k_d",natReg "s_k_t"] [.constNat 0,.constNat 0]

def valuePtr (p : Parameters) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  .makeBlockPtrDynMetadata r (.castIntToNat
    (add (add (mul (reg "i_bh") (reg "s_v_h"))
      (mul (mul (reg "i_t") (integer p.BT)) (reg "s_v_t")))
      (mul (mul (reg "i_v") (integer p.BV)) (reg "s_v_d"))))
    [p.BT,p.BV] [p.BT,p.BV] [natReg "s_v_t",natReg "s_v_d"] [.constNat 0,.constNat 0]

def historyPtr (p : Parameters) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynMetadata r (.castIntToNat
    (add (add (add (mul (reg "i_bh") (reg "s_h_h"))
      (mul (mul (reg "i_t") (integer p.K)) (integer p.V)))
      (mul (mul (reg "i_k") (integer p.BK)) (reg "s_h_t")))
      (mul (mul (reg "i_v") (integer p.BV)) (reg "s_h_d"))))
    [p.BK,p.BV] [p.BK,p.BV] [natReg "s_h_t",natReg "s_h_d"] [.constNat 0,.constNat 0]

def endpointPtr (p : Parameters) (r : RegionName) (head time stride : RegName)
    (block : Nat) (index : RegName) : Op .blockPtr [block] :=
  .makeBlockPtrDynMetadata r (.castIntToNat
    (add (add (mul (reg "i_bh") (reg head))
      (mul (.sub .int .nil (add (mul (reg "i_t") (integer p.BT)) (integer p.BT))
        (.constInt 1)) (reg time)))
      (mul (mul (reg index) (integer block)) (reg stride))))
    [block] [block] [natReg stride] [.constNat 0]

local macro "eval_pointer" : tactic => `(tactic|
  (simp only [keyPtr,valuePtr,historyPtr,endpointPtr,integer,reg,natReg,add,mul,evalOp,
    List.mapM_cons,List.mapM_nil,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.sub,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def,Int.ofNat_eq_natCast]
   simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
   rfl))

theorem key_pointer_eval (p : Parameters) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hk : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hc : u.regs .int [] "i_t"=some (Tile.scalar (c : Int)))
    (hs : StrideRegs p u) :
    evalOp (keyPtr p r) u=some (RebasedAccess.keyPointer p.toLayout l r c) := by
  rcases hs with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9⟩
  simp only [keyPtr,integer,reg,natReg,add,mul,evalOp,hh,hk,hc,h1,h2,h3,
    List.mapM_cons,List.mapM_nil,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def,Int.ofNat_eq_natCast]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rfl

theorem value_pointer_eval (p : Parameters) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hc : u.regs .int [] "i_t"=some (Tile.scalar (c : Int)))
    (hs : StrideRegs p u) :
    evalOp (valuePtr p r) u=some (RebasedAccess.valuePointer p.toLayout l r c) := by
  rcases hs with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9⟩
  simp only [valuePtr,integer,reg,natReg,add,mul,evalOp,hh,hv,hc,h4,h5,h6,
    List.mapM_cons,List.mapM_nil,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def,Int.ofNat_eq_natCast]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rfl

theorem history_pointer_eval (p : Parameters) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hk : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hc : u.regs .int [] "i_t"=some (Tile.scalar (c : Int)))
    (hs : StrideRegs p u) :
    evalOp (historyPtr p r) u=some (RebasedAccess.historyPointer p.toLayout l r c) := by
  rcases hs with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9⟩
  simp only [historyPtr,integer,reg,natReg,add,mul,evalOp,hh,hk,hv,hc,h7,h8,h9,
    List.mapM_cons,List.mapM_nil,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def,Int.ofNat_eq_natCast]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rfl

theorem endpoint_pointer_eval (p : Parameters) (l : Launch) (r : RegionName)
    (c head time stride block indexValue : Nat) (headName timeName strideName index : RegName)
    (u : BlockState) (hBT : 0<p.BT)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hc : u.regs .int [] "i_t"=some (Tile.scalar (c : Int)))
    (hi : u.regs .int [] index=some (Tile.scalar (indexValue : Int)))
    (hhead : u.regs .int [] headName=some (Tile.scalar (head : Int)))
    (htime : u.regs .int [] timeName=some (Tile.scalar (time : Int)))
    (hstride : u.regs .int [] strideName=some (Tile.scalar (stride : Int))) :
    evalOp (endpointPtr p r headName timeName strideName block index) u=
      some (RebasedAccess.endpointPointer p.toLayout l r c head time stride block indexValue) := by
  simp only [endpointPtr,integer,reg,natReg,add,mul,evalOp,hh,hc,hi,hhead,htime,hstride,
    List.mapM_cons,List.mapM_nil,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.sub,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def,Int.ofNat_eq_natCast]
  have hn : ((c*p.BT+p.BT : Nat) : Int)-1=((c*p.BT+p.BT-1 : Nat) : Int) := by omega
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rw [hn]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rfl

#axiomsClean key_pointer_eval
#axiomsClean value_pointer_eval
#axiomsClean history_pointer_eval
#axiomsClean endpoint_pointer_eval
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
