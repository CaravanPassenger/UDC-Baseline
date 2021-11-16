# UDC-Baseline
Baseline code for Under-Display-Camera problem

#### Model

1. UNet (in UNet.py)

   classic deep learning structure 

2. DE_UNet (in DE_UNet.py)

   image restoration model in CVPR_2021_paper: Image Restoration for Under-Display Camera

#### Some Results

TOLED results with UNet trained 400 epochs (Centercrop patches, 256*256)  and DE_UNet trained 400 epochs(same input)

<a href="https://imgtu.com/i/IfEzm6"><img src="https://z3.ax1x.com/2021/11/16/IfEzm6.png" alt="IfEzm6.png" border="0"></a>
<a href="https://imgtu.com/i/IfEj61"><img src="https://z3.ax1x.com/2021/11/16/IfEj61.png" alt="IfEj61.png" border="0"></a>
<a href="https://imgtu.com/i/IfEvOx"><img src="https://z3.ax1x.com/2021/11/16/IfEvOx.png" alt="IfEvOx.png" border="0"></a>

#### Problems to be solved

1. 由于有些输入与label之间的亮度有差距，有些没有，结果呈现出许多亮度高于原图的问题。
2. 可以见到的是网络普遍可以解决blur和noise问题，但是复原问题仍旧不尽人意。

