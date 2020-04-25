# GANとは

画像等を生成させるアルゴリズムの一つ.

画像は$C \subset \mathbb{R}^{n \times n \times c}$とする.

- Genrator
generatorは$z \in \mathbb{R}$?に対し,$G(z) \in C$となる写像.
基本的にはNNで書かれる.

- Discriminator
  $D: C \to  \{0, 1\}$
  画像が生成されたものかどうかで二値分類.


目標はよりよいGeneratorを作成したい.

$x \in C$に対して,確率密度関数を$p_{data}(x)$とする.
つまり,ボレル集合$B \subset C$に対し,$P_{data}(B) = \int_{B} p_{data}(x)dx$となる.
これはデータの分布で$B$に含まれる画像が得られる確率を表す.
同様に$p_g(x)$を定義する.どちらも$x \in C$である.

今$z \mapsto x$がsmoothで,$p_z(z) \frac{\partial x}{\partial z} = p_{g}(x)$を満たすとする.
- 注意:ここで触れていないこと
  - 上の写像がどこからの写像なのか
  - $p_z$が満たすべき仮定
  - $p_g$とGeneratorの関係

この問題では以下となるような$D,G$を求める.

$$
V(D,G):=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))] \\
\min _{G} \max _{D} V(D, G) 
$$

- 厳密さを考えるときには気になること
  - $\log D(x), \log(1- D(G(z))$での期待値が収束するか?(今回は勝手に収束する前提)
  - $\min, \max$ではなく$\sup, \inf$(工学的には存在するのと同等)
  - $p_{data}, p_z$は$z$を指定すると関数ではなく,ただの値になるので,そこからサンプルするという表現はかなり気持ち悪い...
    機械学習で使う確率論の表記はよく理解していないが、意味的には$x \sim p_{data}$なんじゃないだろうか...
  - min,maxの順序を入れ方た時に値は変わるのか?
    - 結局機械学習アルゴリズムではどちらかを固定して最適化を交互にするので、順序の入れ替えで値が変わらないことを期待している.

すぐわかること
- $G$を固定した時,
$$
x \mapsto \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$
となる$D$の時$V(D,G)$は最小.


$$
\begin{aligned}
C(G) &=\max _{D} V(G, D) \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {datata }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_{G}^{*}(G(\boldsymbol{z}))\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(\boldsymbol{x})}{P_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]
\end{aligned}
$$

|Theorem|
|:--|
|$$\min_{G}C(G)$$となるのは$p_g = p_{data}$となるような$g$の時, また$C(G)$の最小値は$- \log 4$|||


なので、学習のときはこれを保つようにしながら,$D$や$G$を学習させるのがポイントだとか...


### DCGAN
Convolutionを使ったGAN

Architecture guidelines for stable Deep Convolutional GANs
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
• Use batchnorm in both the generator and the discriminator.
• Remove fully connected hidden layers for deeper architectures.
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.
• Use LeakyReLU activation in the discriminator for all layers.

これが本質的に聞くかどうかは実験してみて評価.