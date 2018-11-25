# RandomWaypoint
# 1 模型简介

在移动性管理中，随机路点模型是模拟移动用户移动，以及它们的位置，速度和加速度如何随时间变化的随机模型。在评估新的网络协议时，移动性模型用于模拟目的。Random Waypoint Model（RWP）最初由Johnson和Maltz提出，由于其简单性和广泛的可用性，它是评估移动ad hoc网络（MANET）路由协议的最流行的移动模型之一。
在基于随机的移动性仿真模型中，移动节点随机且自由地移动而没有限制。 更具体地说，目的地，速度和方向都是随机选择的，并且与其他节点无关。 这种模型已被用于许多模拟研究中。
RWP存在两种变体：random walk model（RW）和random direction model（RD）。下面介绍RWP和它的两种变体。

## 1.1 Random Waypoint Model

在RWP中，初始状态时，结点在整个仿真区域内服从均匀分布，结点首先从二维仿真区域中随机选择一个结点作为目的地，然后从[V~min~, V~max~]中随机选择一个速度（服从均匀分布），结点以此速度向目的地运动。在到达目的地后，结点在[0, P~max~]中随机选择一段停留时间T，然后选择下一个目的地。RWP中结点运动模式如图1.1.1所示。
 <div align=center><img  src="https://img-blog.csdnimg.cn/20181123222834361.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
 
<center>图1.1.1 RWP结点运动模式</center>

另外，RWP中存在密度波(density wave)的现象，具体来说就是结点会随着时间的推移表现出非均匀分布，在仿真区域的中心处达到最大，而在边界处密度趋于0。下面是论文[1]中描述RWP密度波现象的图。

 <div align=center><img  src="https://img-blog.csdnimg.cn/20181125120414946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
 
<center>图1.1.2 density wave</center>

在后面利用RWP生成室内轨迹数据也可以看到这种现象。

## 1.2 Random Walk Model
RW作为RWP的变体，也是一种重要的随机移动性模型。RW中结点从[0, 2π]随机的选择一个方向，从[V~min~, V~max~]随机选择一个速度，然后按照选取的方向和速度移动到新的位置。在结点移动的过程中，选择一个时间间隔t或者固定距离d，当结点运动了t时间或者移动了d长度时，重新选择结点的方向和速度。RW中结点的运动模式如图1.2.1所示。

 <div align=center><img  src="https://img-blog.csdnimg.cn/20181125120604983.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
 
<center>图1.2.1 RW结点运动模式</center>

当结点到达仿真区域的边界时，要根据当前结点运动的方向以一定的角度从边界弹回。

## 1.3 Random Direction Model
在RD中，结点随机地从[0, 2π]选择一个方向，然后按此方向一直移动，直到达到仿真区域的边界，在[0, P~max~]中随机选择一段停留时间T，再从[0, π]之间选择一个角度，继续移动。RD可以克服RWP引起的density wave现象。

 <div align=center><img  src="https://img-blog.csdnimg.cn/20181125115845121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
 
<center>图1.3.1 RD中结点的移动模式</center>

# 2 生成轨迹数据

基于RWP，就可以模拟行人的移动：行人按一定的步长和方向移动到目的地，暂停一段时间，然后改变方向，按一定的步长移动到另一个目的地。
由于之前需要生成轨迹数据训练RNN，因此博主编写了基于RWP生成室内行人运动轨迹数据的程序，图2.1是总步数为2000的行人运动轨迹。

  <div align=center><img  src="https://img-blog.csdnimg.cn/20181123225232648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
  
<center>图2.1 2000step RWP</center>

前面提到，RWP存在density wave现象，导致均匀分布的空间点随着时间累积，会转换成非均匀分布，最终导致中间密度大，边缘密度趋紧于0，在图2.1中可以明显观察到这种现象。
可以通过RD对轨迹生成模型进行改进，RD是RWP模型的变体，将RD模型引入轨迹生成模型，在生成轨迹时有概率基于RD生成轨迹，改进后的模型生成的行人运动轨迹如图2.2。
 <div align=center><img  src="https://img-blog.csdnimg.cn/20181125111509856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTc0NzE5,size_16,color_FFFFFF,t_70"/></div>
 
<center>图2.2 2000step RWP-RP（RP pro=0.3）</center>

# 链接
Wiki https://en.wikipedia.org/wiki/Random_waypoint_model</br>
论文[1] Visualization of Spatial Distribution of Random Waypoint Mobility Models http://www.jcomputers.us/vol12/jcp1204-04.pdf
