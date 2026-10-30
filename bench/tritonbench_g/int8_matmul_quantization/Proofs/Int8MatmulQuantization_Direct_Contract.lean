import bench.tritonbench_g.int8_matmul_quantization.Proofs.Contract
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Index
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

theorem tensorContract_iff (t : TensorInfo) (types : List VeriTile.Torch.DType) (rank : Nat) (column : Bool) :
    TensorContract t types rank column ↔ t.metadata.layout=.strided ∧ t.metadata.shape.length=rank ∧
      t.metadata.dtype∈types ∧ t.metadata.device.kind="cuda" ∧ t.metadata.isNeg=Bool.false ∧
      (t.metadata.isContiguous=Bool.true ∨ column=Bool.true ∧ t.transposeContiguous=Bool.true) ∧
      (0<t.metadata.numel → t.metadata.dataPtr.toNat%bytes t=0) := by
  constructor
  · intro h;exact ⟨h.layout,h.rank,h.dtype,h.cuda,h.noNeg,h.dense,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g⟩;exact ⟨a,b,c,d,e,f,g⟩
theorem tensorValid_iff (t : TensorInfo) (types : List VeriTile.Torch.DType) (rank : Nat) (column : Bool) :
    tensorValid t types rank column=Bool.true ↔ TensorContract t types rank column := by
  rw [tensorContract_iff]
  simp [tensorValid,VeriTile.Torch.TensorMetadata.isCuda,and_assoc]
  omega

theorem rowContract_iff (t : TensorInfo) : RowContract t ↔
    TensorContract t [.floating .fp32] 2 Bool.false ∧
    (rows t=0 ∨ (0<cols t ∧ cols t≤2^20 ∧ rows t≤2^31-1 ∧
      4*(t.offset+rows t*max (cols t) 1+2*width (cols t))<2^63)) := by
  exact ⟨fun h=>⟨h.input,h.launch⟩,fun ⟨a,b⟩=>⟨a,b⟩⟩
theorem rowValid_iff (t : TensorInfo) : rowValid t=Bool.true ↔ RowContract t := by
  rw [rowContract_iff]
  simp [rowValid,tensorValid_iff,and_assoc]

theorem tileContract_iff (a b bs : TensorInfo) (as out : Option TensorInfo) (composed : Bool) (c : Tile) :
    TileContract a b bs as out composed c ↔
    (rows a+c.BM<2^63 ∧ cols b+c.BN<2^63 ∧ cols a+c.BK*c.S<2^63) ∧
    (gm a c*c.BM<2^63 ∧ gn b c*c.BN<2^63 ∧ fk a c<2^63) ∧ c.G*gn b c<2^63 ∧
    (gm a c*gn b c≤2^31-1 ∧ c.S≤65535) ∧
    (if composed then 0 else a.offset)+rows a*max (cols a) 1+fk a c<2^63 ∧
    b.offset+fk a c*bRowStride a b+cols b*bColStride a b<2^63 ∧
    outputBytes out*(offset out+gm a c*c.BM*max (cols b) 1+gn b c*c.BN)<2^63 ∧
    4*((if composed then 0 else offset as)+rows a)<2^63 ∧ 4*(bs.offset+cols b)<2^63 := by
  constructor
  · intro h;exact ⟨h.dimensions,h.coordinates,h.groupWidth,h.grid,h.aFormed,h.bFormed,h.outFormed,h.aScaleFormed,h.bScaleFormed⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩
theorem tileValid_iff (a b bs : TensorInfo) (as out : Option TensorInfo) (composed : Bool) (c : Tile) :
    tileValid a b bs as out composed c=Bool.true ↔ TileContract a b bs as out composed c := by
  rw [tileContract_iff]
  simp [tileValid,and_assoc]

theorem outputContract_iff (a b : TensorInfo) (ts : List TensorInfo) (out : TensorInfo) :
    OutputContract a b ts out ↔ TensorContract out [.other "float16",.floating .fp32] 2 Bool.false ∧
      out.metadata.shape=[rows a,cols b] ∧ out.metadata.device=a.metadata.device ∧ ∀ t∈ts,out.storage≠t.storage := by
  exact ⟨fun h=>⟨h.tensor,h.shape,h.device,h.separated⟩,fun ⟨a,b,c,d⟩=>⟨a,b,c,d⟩⟩
theorem outputValid_iff (a b : TensorInfo) (ts : List TensorInfo) (out : TensorInfo) :
    outputValid a b ts out=Bool.true ↔ OutputContract a b ts out := by
  rw [outputContract_iff]
  simp [outputValid,tensorValid_iff,and_assoc]

theorem gemmContract_iff (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo) (composed : Bool) :
    GemmContract a as b bs out composed ↔
    TensorContract a [if composed then .floating .fp32 else .other "int8"] 2 Bool.false ∧
    TensorContract b [.other "int8"] 2 Bool.true ∧ TensorContract bs [.floating .fp32] 1 Bool.false ∧
    (composed=Bool.false → ∃ t,as=some t ∧ TensorContract t [.other "float16",.floating .fp32] 1 Bool.false ∧ t.metadata.shape=[rows a]) ∧
    (rows b=cols a ∧ bs.metadata.shape=[cols b]) ∧
    (∀ t∈inputs a as b bs composed,t.metadata.device=a.metadata.device) ∧ MixedSeparate (inputs a as b bs composed) ∧
    (∀ t,out=some t → OutputContract a b (inputs a as b bs composed) t) ∧
    (rows a=0 ∨ cols b=0 ∨ ∀ c∈tiles,TileContract a b bs as out composed c) := by
  constructor
  · intro h;exact ⟨h.aTensor,h.bTensor,h.bScaleTensor,h.aScale,h.shape,h.device,h.mixedAlias,h.output,h.launch⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩
theorem gemmValid_iff (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo) (composed : Bool) :
    gemmValid a as b bs out composed=Bool.true ↔ GemmContract a as b bs out composed := by
  rw [gemmContract_iff]
  cases as <;> cases out <;> cases composed <;>
    simp [gemmValid,tensorValid_iff,outputValid_iff,tileValid_iff,MixedSeparate,and_assoc,or_assoc]

theorem helperContract_iff (t : TensorInfo) (ax : Int) : HelperContract t ax ↔
    TensorContract t [.floating .fp32] 2 Bool.false ∧ (ax=0 ∨ ax=1 ∨ ax= -1 ∨ ax= -2) ∧
    0<(if ax=0 ∨ ax= -2 then rows t else cols t) := by
  exact ⟨fun h=>⟨h.input,h.axes,h.nonempty⟩,fun ⟨a,b,c⟩=>⟨a,b,c⟩⟩
theorem helperValid_iff (t : TensorInfo) (ax : Int) : helperValid t ax=Bool.true ↔ HelperContract t ax := by
  rw [helperContract_iff]
  simp [helperValid,tensorValid_iff,and_assoc,or_assoc]

theorem checkRow_iff (a : Argument) : checkRow a=.ok () ↔
    match a with | .tensor t=>RowContract t | _=>False := by
  cases a <;> simp [checkRow,rowValid_iff]
theorem checkGemm_iff (a as b bs : Argument) (out : OptionalArgument) : checkGemm a as b bs out=.ok () ↔
    match a,as,b,bs,out with
    | .tensor a,.tensor as,.tensor b,.tensor bs,.none=>GemmContract a (some as) b bs none Bool.false
    | .tensor a,.tensor as,.tensor b,.tensor bs,.tensor out=>GemmContract a (some as) b bs (some out) Bool.false
    | _,_,_,_,_=>False := by
  cases a <;> cases as <;> cases b <;> cases bs <;> cases out <;> simp [checkGemm,gemmValid_iff]
theorem checkComposed_iff (a b bs : Argument) (out : OptionalArgument) : checkComposed a b bs out=.ok () ↔
    match a,b,bs,out with
    | .tensor a,.tensor b,.tensor bs,.none=>ComposedContract a b bs none
    | .tensor a,.tensor b,.tensor bs,.tensor out=>ComposedContract a b bs (some out)
    | _,_,_,_=>False := by
  cases a <;> cases b <;> cases bs <;> cases out <;> simp [checkComposed,ComposedContract,rowValid_iff,gemmValid_iff]
theorem checkHelper_iff (a : Argument) (ax : AxisArgument) : checkHelper a ax=.ok () ↔
    match a,ax with | .tensor a,.integer ax=>HelperContract a ax | _,_=>False := by
  cases a <;> cases ax <;> simp [checkHelper,helperValid_iff]

def sourceTile (c : Config) : Tile := ⟨c.BM,c.BN,c.BK,c.S,c.group⟩
theorem tiles_match : tiles=Source.configs.map sourceTile := rfl

def shape (a b : TensorInfo) : Shape := ⟨rows a,cols b,cols a⟩
def scalePrecision (as : Option TensorInfo) (composed : Bool) : Precision :=
  if composed then .half else if (as.map (fun t=>t.metadata.dtype)).getD (.other "float16")=.floating .fp32 then .single else .half

def gemmBuffers (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo)
    (composed : Bool) (qa qs output : Nat) : Buffers where
  a := if composed then qa else a.storage
  b := b.storage
  aScale := if composed then qs else (as.map TensorInfo.storage).getD qs
  bScale := bs.storage
  out := output
  aOffset := if composed then 0 else a.offset
  bOffset := b.offset
  aScaleOffset := if composed then 0 else offset as
  bScaleOffset := bs.offset
  outOffset := offset out
  bColumnMajor := columnMajor b
  aScalePrecision := scalePrecision as composed
  bScalePrecision := .single
  outPrecision := if singleOutput out then .single else .half

theorem nativeGemm (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo)
    (composed : Bool) (qa qs output : Nat) (h : GemmContract a as b bs out composed)
    (hm : 0<rows a) (hn : 0<cols b) (c : Config) (member : Source.Admitted c) :
    IndexABI (shape a b) c (gemmBuffers a as b bs out composed qa qs output) := by
  have mem : sourceTile c∈tiles := by rw [tiles_match];exact List.mem_map.mpr ⟨c,member,rfl⟩
  have tile : TileContract a b bs as out composed (sourceTile c) := by
    rcases h.launch with z|z|all <;> try omega
    exact all _ mem
  refine ⟨tile.dimensions,tile.coordinates,tile.groupWidth,tile.grid.1,tile.grid.2,?_,?_,?_,tile.aScaleFormed,tile.bScaleFormed⟩
  · exact ⟨by change (0:Nat)<1;decide,by simpa [aRectangle,gemmBuffers,shape,formedK,Int8DequantMatmul.formedK,Int8DequantMatmul.steps,fk,sourceTile] using tile.aFormed⟩
  · exact ⟨by change (0:Nat)<1;decide,by simpa [bRectangle,gemmBuffers,shape,formedK,Int8DequantMatmul.formedK,Int8DequantMatmul.steps,fk,sourceTile,bRowStride,bColStride] using tile.bFormed⟩
  · constructor
    · cases ho : singleOutput out <;> simp [outRectangle,gemmBuffers,ho]
    · cases ho : singleOutput out <;> simpa [outputBytes,outRectangle,gemmBuffers,shape,padM,padN,Int8DequantMatmul.padM,Int8DequantMatmul.padN,gm,gn,sourceTile,ho] using tile.outFormed

def rowBuffers (input : TensorInfo) (data scales : Nat) : QuantBuffers :=
  ⟨input.storage,data,scales,input.offset,0,0,max (cols input) 1,1,Bool.false,.half,Bool.true⟩
theorem nativeRow (input : TensorInfo) (data scales : Nat) (h : RowContract input) (positive : 0<rows input) :
    QuantIndexABI (rows input) (cols input) (width (cols input)) (rowBuffers input data scales) := by
  have launch : 0<cols input ∧ cols input≤2^20 ∧ rows input≤2^31-1 ∧
      4*(input.offset+rows input*max (cols input) 1+2*width (cols input))<2^63 := by
    rcases h.launch with zero|hh <;> omega
  have widthPositive : 0<width (cols input) := Nat.pow_pos (by decide)
  have widthCover : cols input≤width (cols input) := Nat.le_pow_clog (by decide) _
  have widthBound : width (cols input)≤2^20 := Nat.pow_le_pow_right (by decide) (Nat.clog_le_of_le_pow launch.2.1)
  have maxeq : max (cols input) 1=cols input := max_eq_left (by omega)
  refine ⟨widthPositive,widthCover,Or.inr launch.1,launch.2.2.1,?_,?_,?_,?_⟩
  · omega
  · exact ⟨by change (0:Nat)<4;decide,by simpa [rowBuffers] using launch.2.2.2⟩
  · change 1*(0+rows input*cols input+2*width (cols input))<2^63
    rw [maxeq] at launch
    omega
  · change 4*(0+rows input)<2^63;omega

#axiomsClean checkRow_iff
#axiomsClean checkGemm_iff
#axiomsClean checkComposed_iff
#axiomsClean checkHelper_iff
#axiomsClean nativeGemm
#axiomsClean nativeRow
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
