[^1]

**Index Terms**: simultaneous speech translation, direct speech
translation, attention, alignment

# Introduction

Simultaneous speech translation (SimulST) involves the generation, with
minimal delay, of partial translations for an incrementally received
input audio. In the quest for high-quality output and low latency,
recent developments led to the advent of direct methods, which have been
demonstrated to outperform the traditional cascaded (ASR + MT) pipelines
in terms of both quality and latency
[@anastasopoulos-etal-2022-findings]. Early works on direct SimulST
require the training of several models which were optimized for
different latency regimes
[@ren-etal-2020-simulspeech; @ma-etal-2020-simulmt; @zeng-etal-2021-realtrans],
consequently resulting in high computational and maintenance costs. With
the aim of reducing this computational burden, the use of
offline-trained direct ST models for the simultaneous inference has been
recently studied [@papi-etal-2022-simultaneous] and is becoming popular
[@liu20s_interspeech; @chen-etal-2021-direct; @nguyen2021empirical] due
to its competitive performance compared to dedicated architectures
specifically developed for SimulST [@anastasopoulos-etal-2022-findings].
Indeed, this approach enables an offline ST model to work in
simultaneous by applying, only at inference time, a so-called *decision
policy*, which is in charge to determine whether to emit a partial
hypothesis or wait for more audio input. As a result, no specific
adaptation is required either for the SimulST task or to achieve
different latency regimes.

Along this line of research, we propose [AlignAtt]{.smallcaps}, a novel
policy for SimulST that exploits the audio-translation alignments
obtained from the attention weights of an offline-trained model to
decide whether to emit or not a partial translation. Our policy is based
on the idea that, if the candidate token is aligned with the last frames
of the input audio, the information encoded can be insufficient to
safely produce that token. The audio-translation alignments are
automatically generated from the attention weights, whose
representativeness has been extensively studied in linguistics-related
tasks
[@raganato-tiedemann-2018-analysis; @htut2019attention; @Lamarre2022],
including word-alignment in machine translation
[@tang-etal-2018-analysis; @garg-etal-2019-jointly; @chen-etal-2020-accurate].

All in all, the contributions of our work are the following:

-   We present [AlignAtt]{.smallcaps}, a novel decision policy for
    SimulST that guides an offline-trained model during simultaneous
    inference by leveraging audio-translation alignments computed from
    the attention weights;

-   We compare [AlignAtt]{.smallcaps} with popular and state-of-the-art
    policies that can be applied to offline-trained ST models, achieving
    the new state of the art on all the 8 languages of MuST-C v1.0
    [@CATTONI2021101155], with gains of 2 BLEU points and a latency
    reduction of 0.5-0.8$s$ depending on the target languages;

-   The code, the models, and the simultaneous outputs are published
    under Apache 2.0 Licence at:
    <https://github.com/hlt-mt/fbk-fairseq>.

# [AlignAtt]{.smallcaps} policy {#sec:policy}

[AlignAtt]{.smallcaps} is based on the source audio - target text
alignment obtained through the attention scores of a Transformer-based
model [@transformer]. In the Transformer, encoder-decoder (or cross)
attention $A_C$ is computed by applying the standard dot-product
mechanism [@7472621] as follows:
$$A_C(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$ where
the matrices $K$ (key) and $V$ (value) are obtained from the encoder
output and consequently depend on the input source $\mathbf{x}$, the
matrix $Q$ (query) is obtained from the output of the previous decoder
layer (or from the previous output tokens in case of the first decoder
layer), and consequently depends on the prediction $\mathbf{y}$, and
$d_k$ is a scaling factor. Cross attention can be hence expressed as a
function of $\mathbf{x}$ and $\mathbf{y}$, obtaining
$A_C(\mathbf{x}, \mathbf{y})$. Exploiting the cross attention
$A_C(\mathbf{x}, \mathbf{y})$, the alignment vector $Align$ is computed
by considering, for each token $y_i$ of the prediction
$\mathbf{y}=[y_1,...,y_m]$, the index of the most attended frame (or
encoder state) $x_j$ of the source input $\mathbf{x}=[x_1,...,x_n]$:
$$Align_i = \arg \max_{j}  A_C(\mathbf{x}, y_i)$$ This means that, for
every predicted token $y_i$, we have a unique aligned frame $x_j$ of
index $Align_i$.

Our policy (Figure [1](#fig:alignatt){reference-type="ref"
reference="fig:alignatt"}) exploits the obtained alignment $Align$ to
guide the model during inference by checking whether each token $y_i$
attends to the last $f$ frames or not. If this condition is verified,
the emission is stopped, under the assumption that, if a token is
aligned with the most recently received audio frames, the information
they provide can be insufficient to generate that token (i.e. the system
has to wait for additional audio input). Specifically, starting from the
first token, we iterate over the prediction $\mathbf{y}$ and continue
the emission until: $$Align_i \notin \{n-f+1, ..., n\}$$ which means
that we stop the emission as soon as we find a token that mostly attends
to one of the last $f$ frames. Thus, $f$ is the parameter that directly
controls the latency of the model: smaller $f$ values mean fewer frames
to be considered inaccessible by the model, consequently implying a
lower chance that our stopping condition is verified and, in turn, lower
latency. The process is formalized in Algorithm
[\[alg:alignatt\]](#alg:alignatt){reference-type="ref"
reference="alg:alignatt"}.

:::: algorithm
::: algorithmic
$Align$, $f$, $\mathbf{y}$ $i \gets 1$ $prediction \gets [\quad]$
$stop \gets False$ $stop \gets True$ $prediction \gets prediction + y_i$
$i \gets i + 1$
:::
::::

Since in SimulST the source speech input $\mathbf{x}$ is incrementally
received and its length $n$ is increased at every time step $t$,
applying the [AlignAtt]{.smallcaps} policy means applying Algorithm
[\[alg:alignatt\]](#alg:alignatt){reference-type="ref"
reference="alg:alignatt"} at each timestep to emit (or not) the partial
hypothesis until the input $\mathbf{x}(t)$ has been entirely received.

# Experimental Settings

## Data {#subsec:data}

We train one model for each of the 8 languages of MuST-C
v1.0 [@CATTONI2021101155], namely English (en) to Dutch (nl), French
(fr), German (de), Italian (it), Portuguese (pt), Romanian (ro), Russian
(ru), and Spanish (es). We filter out segments longer than 30$s$ from
the training set to optimize GPU RAM consumption. We also apply
sequence-level knowledge distillation [@kim2016sequencelevel] to
increase the size of our training set and improve performance. To this
aim, we employ NLLB 3.3B [@costa2022no] as the MT model to translate the
English transcripts of the training set into each of the 8 languages,
and we use the automatic translations together with the gold ones during
training. As a result, the final number of target sentences is twice the
original one while the speech input remains unaltered. The performance
of the NLLB 3.3B model on the MuST-C v1.0 test set is shown in Table
[1](#tab:nllb){reference-type="ref" reference="tab:nllb"}.

::: {#tab:nllb}
  Model     de     es     fr     it     nl     pt     ro     ru    Avg
  ------- ------ ------ ------ ------ ------ ------ ------ ------ ------
  NLLB     33.1   38.5   46.5   34.4   37.7   40.4   32.8   23.5   35.9
                                                                  

  : BLEU results on all the language pairs of MuST-C v1.0 tst-COMMON of
  NLLB 3.3B model.
:::

<figure id="fig:alignatt">

<figcaption>Example of the <span class="smallcaps">AlignAtt</span>
policy with <span class="math inline"><em>f</em> = 2</span> at
consecutive time steps <span
class="math inline"><em>t</em><sub>1</sub></span> (a) and <span
class="math inline"><em>t</em><sub>2</sub></span> (b).</figcaption>
</figure>

## Architecture and Training Setup {#sec:architecture}

The model is made of 12 Conformer [@gulati20_interspeech] encoder layers
and 6 Transformer decoder layers, having 8 attention heads each. The
embedding size is set to 512 and the feed-forward layers are composed of
2,048 neurons, with $\sim$`<!-- -->`{=html}115M parameters in total. The
input is represented by 80 log Mel-filterbank audio features extracted
every 10$ms$ with a sample window of 25, and pre-processed by two 1D
convolutional layers of striding 2 to reduce the input length by a
factor of 4 [@wang2020fairseqs2t]. Dropout is set to 0.1 for attention,
feed-forward, and convolutional layers. The kernel size is 31 for both
point- and depth-wise convolutions in the Conformer encoder. The
SentencePiece-based [@sennrich-etal-2016-neural] vocabulary size is
8,000 for translation and 5,000 for transcript. Adam optimizer with
label-smoothed cross-entropy loss (smoothing factor 0.1) is used during
training together with CTC loss [@Graves2006ConnectionistTC] to compress
audio input representation and speed-up inference
time [@gaido-etal-2021-ctc]. Learning rate is set to $5\cdot10^{-3}$
with Noam scheduler and 25,000 warm-up steps. Utterance-level Cepstral
Mean and Variance Normalization (CMVN) and SpecAugment [@Park2019] are
also applied during training. Trainings are performed on 2 NVIDIA A40
GPUs with 40GB RAM. We set 40k as the maximum number of tokens per
mini-batch, update frequency 4, and 100,000 maximum updates
($\sim$`<!-- -->`{=html}28 hours). Early stopping is applied during
training if validation loss does not improve for 10 epochs. We use the
bug-free implementation of fairseq-ST [@papi2023reproducibility].

## Terms of Comparison {#subsec:comparison}

We conduct experimental comparisons with the other SimulST policies that
can be applied to offline systems, thus policies that do not require
training nor adaptation to be run, namely:

-   **Local Agreement (LA)** [@liu20s_interspeech]: the policy used by
    [@polak-etal-2022-cuni] to win the SimulST task at the IWSLT 2022
    evaluation campaign [@anastasopoulos-etal-2022-findings]. With this
    policy, a partial hypothesis is generated each time a new speech
    segment is added as input, and it is emitted, entirely or partially,
    if the previously generated hypothesis is equal to the current one.
    We adapted the docker released by the authors to Fairseq-ST
    [@wang2020fairseqs2t]. Different latency regimes are obtained by
    varying the speech segment length $T_s$.

-   **Wait-k** [@ma-etal-2019-stacl]: the most popular policy originally
    published for simultaneous machine translation and then adapted to
    SimulST [@ren-etal-2020-simulspeech; @zeng-etal-2021-realtrans]. It
    consists in waiting for a predefined number of words ($k$) before
    starting to alternate between writing a word and waiting for new
    output. We employ adaptive word detection guided by the CTC
    prediction to detect the number of words in the speech as in
    [@zeng-etal-2021-realtrans; @papi-etal-2022-simultaneous].

-   **[EDAtt]{.smallcaps}** [@papi2022attention]: the only existing
    policy that exploits the attention mechanism to guide the inference.
    Contrary to our policy that computes audio-text alignments starting
    from the attention scores, in [EDAtt]{.smallcaps} the attention
    scores of the last $\lambda$ frames are summed and a threshold
    $\alpha$ is used to trigger the emission. While $\alpha$ handles the
    latency, $\lambda$ is a hyper-parameter that has to be empirically
    determined on the validation set. This represents the main flaw of
    this policy since, in theory, $\lambda$ has to be estimated for each
    language. Here, we set $\lambda=2$ following the authors' finding.

## Inference and Evaluation

For inference, the input features are computed on the fly and Global
CMVN normalization is applied as in [@ma-etal-2020-simulmt]. We use the
SimulEval tool [@ma-etal-2020-simuleval] to compare
[AlignAtt]{.smallcaps} with the above policies. For the LA policy, we
set $T_s=[10,15,20,25,30]$[^2]; for the wait-k, we vary $k$ in
$[2,3,4,5,6,7]$[^3]; for [EDAtt]{.smallcaps}, we set
$\alpha=[0.6,0.4,0.2,0.1,0.05,0.03]$[^4]; for [AlignAtt]{.smallcaps}, we
vary $f$ in $[2,4,6,8,10,12,14]$. Moreover, to be comparable with
[EDAtt]{.smallcaps}, for our policy we extract the attention weights
from the 4^th^ decoder layer and average across all the attention heads.
All inferences are performed on a single NVIDIA TESLA K80 GPU with 12GB
of RAM as in the IWSLT Simultaneous evaluation campaigns
[@iwslt_2021; @anastasopoulos-etal-2022-findings]. We use sacreBLEU
($\uparrow$) [@post-2018-call][^5] to evaluate translation quality and
Length Adaptive Average Lagging [@papi-etal-2022-generation] -- or LAAL
($\downarrow$) -- to measure latency.[^6] As suggested by
[@ma-etal-2020-simulmt], we report the computational-aware version of
LAAL[^7] that accounts for the real elapsed time instead of the ideal
one, consequently providing a more realistic latency measure.

# Results {#sec:exps}

In this section, we present the results of our offline systems trained
for each language pair of MuST-C v1.0 to show their competitiveness
compared to the systems published in literature (Section
[4.1](#subsec:offline_res){reference-type="ref"
reference="subsec:offline_res"}) and the results of the
[AlignAtt]{.smallcaps} policy compared to the other policies presented
in Section [3.3](#subsec:comparison){reference-type="ref"
reference="subsec:comparison"} (Section
[4.2](#subsec:simul_res){reference-type="ref"
reference="subsec:simul_res"}).

## Offline Results {#subsec:offline_res}

To provide an upper bound to the simultaneous performance and show the
competitiveness of our models, we present in Table
[\[tab:offline_res\]](#tab:offline_res){reference-type="ref"
reference="tab:offline_res"} the offline results of the systems trained
on all the language pairs of MuST-C v1.0 compared to systems published
in literature that report results for all languages. As we can see, our
offline systems outperform the others on all but 2 language pairs,
en$\rightarrow${es, fr, it, nl, pt, ro}, achieving the new state of the
art in terms of translation quality. BLEU gains are more evident for
en$\rightarrow$fr and en$\rightarrow$it, for which we obtain
improvements of about 1 BLEU point, while they amount to about 0.5 BLEU
points for the other languages.

Concerning the other 2 languages (de, ru), our en$\rightarrow$ru model
achieves a similar result (18.4 vs 18.5 BLEU) with that obtained by the
best model for that language (XSTNet [@ye21_interspeech]), with only a
0.1 BLEU drop. Moreover, our system reaches a slightly worse but
competitive result for en$\rightarrow$de (28.0 vs 28.7 BLEU) compared to
STEMM [@fang-etal-2022-stemm], which instead makes use of a relevant
amount of external speech data, and it also outperforms all the other
systems for this language direction. On average, our approach stands out
as the best one even if it does not involve the use of external speech
data[:]{style="color: black"} it obtains an average of 29.4 BLEU across
languages, which corresponds to 0.5 to 4.6 BLEU improvements compared to
the published ST models.

## Simultaneous Results {#subsec:simul_res}

Having demonstrated the competitiveness of our offline models, we now
apply the SimulST policies introduced in Section
[3.3](#subsec:comparison){reference-type="ref"
reference="subsec:comparison"} to the same offline ST model for each
language pair of MuST-C v1.0. Figure
[\[fig:simul_res\]](#fig:simul_res){reference-type="ref"
reference="fig:simul_res"} shows the results in terms of latency-quality
trade-off (i.e. LAAL ($\downarrow$) - BLEU ($\uparrow$) curves).

As we can see, our [AlignAtt]{.smallcaps} policy is the only policy,
together with [EDAtt]{.smallcaps}, capable of reaching a latency lower
or equal to 2$s$ for all the 8 languages.[^8] Specifically, LA curves
start at around 2.5$s$ or more for all the language pairs, even if they
are able to achieve high translation quality towards 3.5$s$, with a 1.2
average drop in terms of BLEU across languages compared to the offline
inference. Similarly, the wait-k curves start at around 2/2.5$s$ but are
not able to reach high translation quality even at high latency (LAAL
approaching 3.5$s$), therefore scoring the worst results. Compared to
these two policies, [AlignAtt]{.smallcaps} shows a LAAL reduction of up
to 0.8$s$ compared to LA and 0.5$s$ compared to wait-k. Despite
achieving lower latency as [AlignAtt]{.smallcaps}, the
[EDAtt]{.smallcaps} policy achieves worse translation quality at almost
every latency regime compared to our policy, with drops of up to 2 BLEU
points across languages. These performance drops are particularly
evident for en$\rightarrow$de and en$\rightarrow$ru, where the latter
represents the most difficult language pair also in offline ST (it is
the only language with less than 20 BLEU on Table
[\[tab:offline_res\]](#tab:offline_res){reference-type="ref"
reference="tab:offline_res"}). The evident differences in the
[AlignAtt]{.smallcaps} and [EDAtt]{.smallcaps} policy behaviors,
especially in terms of translation quality, prove that, despite both
exploiting attention scores as a source of information, the decisions
taken by the two policies are intrinsically different. Moreover,
[AlignAtt]{.smallcaps} is the closest policy to achieving the offline
results of Table
[\[tab:offline_res\]](#tab:offline_res){reference-type="ref"
reference="tab:offline_res"}, with less than 1.0 BLEU average drop
versus 1.8 of [EDAtt]{.smallcaps}.

We can conclude that, on all the 8 languages of MuST-C v1.0, the
[AlignAtt]{.smallcaps} policy achieves a lower latency compared to both
wait-k and LA, and an improved translation quality compared to
[EDAtt]{.smallcaps}, therefore representing the new state-of-the-art
SimulST policy applicable to offline ST models.

# Conclusions

We presented [AlignAtt]{.smallcaps}, a novel policy for SimulST that
leverages the audio-translation alignments obtained from the
cross-attention scores to guide an offline-trained ST model during
simultaneous inference. Results on all 8 languages of MuST-C v1.0 showed
the effectiveness of our policy compared to the existing ones, with
gains of 2 BLEU and a latency reduction of 0.5-0.8$s$, achieving the new
state of the art. Code, offline ST models, and simultaneous outputs are
released open source to help the reproducibility of our work.

[^1]: We acknowledge the support of the PNRR project FAIR - Future AI
    Research (PE00000013), under the NRRP MUR program funded by the
    NextGenerationEU.

[^2]: Smaller values of $T_s$ do not improve computational aware
    latency.

[^3]: We do not report results obtained with $k=1$ since the translation
    quality highly degrades.

[^4]: These are the same values indicated by the authors of the policy.

[^5]: BLEU+case.mixed+smooth.exp+tok.13a+version.1.5.1

[^6]: Length Adaptive Average Lagging is a an improved speech version of
    Average Lagging [@ma-etal-2019-stacl], which accounts for both
    longer and shorter predictions compared to the reference.

[^7]: We present all the results with
    $\text{LAAL\textsubscript{max}}=3.5s$.

[^8]: The maximum acceptable latency limit is set between 2$s$ and 3$s$
    from most works on simultaneous interpretation
    [@doi:10.1177/002383097501800310; @fantinuoli2022defining].
