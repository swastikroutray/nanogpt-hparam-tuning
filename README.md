**1. Introduction**

Transformer-based language models have become the foundation of modern Natural Language Processing (NLP), powering systems such as GPT, Claude, Gemini, and Llama. 
Although the Transformer architecture remains largely consistent, its performance depends heavily on selecting appropriate hyperparameters. 
Hyperparameters such as learning rate, dropout, embedding dimension, number of transformer layers, batch size, and context window significantly 
affect convergence speed, training stability, computational efficiency, and generalization.

The objective of this study was to systematically investigate how different hyperparameters influence the training and validation performance of a Transformer 
language model trained on War and Peace. Rather than modifying the architecture itself, this work focuses on empirical hyperparameter optimization 
by varying one parameter at a time while keeping the remaining parameters fixed.

The primary evaluation metric used throughout the experiments was cross-entropy loss, measured on both the training and validation datasets. 
Lower validation loss indicates better generalization to unseen text, while the difference between training and validation loss provides insight 
into possible overfitting or underfitting.

**2. Methodology**

**2.1 Model**

The experiments were conducted using a decoder-only Transformer language model implemented in PyTorch, 
following the nanoGPT architecture introduced by Andrej Karpathy.

The model consists of:

1) Token Embedding Layer
2) Positional Embeddings
3) Multi-Head Self-Attention
4) Feed-Forward Networks
5) Residual Connections
6) Layer Normalization
7) Linear Output Head

The model was trained using next-token prediction with the Cross Entropy Loss.

**2.2 Dataset**

The model was trained on War and Peace, where each character acts as an individual token after encoding.

The dataset was divided into:

1) Training Set
2) Validation Set

***Validation loss was used as the primary metric for selecting optimal hyperparameters.***

**2.3 Experimental Procedure**

Each experiment modified only one hyperparameter while keeping all others constant. 
This isolates the effect of each parameter on model performance.

The experiments investigated:

1) Learning Rate
2) Dropout Rate
3) Number of Transformer Layers
4) Embedding Dimension
5) Batch Size
6) Block Size (Context Length)
7) Final Dropout Verification

Training loss and validation loss after convergence were recorded for every configuration.

**3. Results**
**3.1 Effect of Learning Rate**
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/ca6da71b-350f-4faf-aa25-77cf1519c7f9" />

Learning rate was swept across 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, and 3e-2. Performance follows a classic U-shaped curve. 
At very low learning rates (1e-5), both training and validation loss are high (~2.75 and ~2.73, respectively), indicating the model 
is undertrained — it simply hasn't moved far enough from initialization. Loss improves steadily as the learning rate increases, 
reaching a minimum around 3e-3, where training loss drops to roughly 1.56 and validation loss to roughly 1.65. Beyond this point, 
performance degrades sharply: at 1e-2 validation loss rises to ~1.76, and at 3e-2 the model destabilizes entirely, with both losses 
jumping back up to ~2.7. This is consistent with the learning rate becoming too large for stable gradient descent, causing the optimizer 
to overshoot minima. The gap between training and validation loss stays small and roughly constant across the sweep, suggesting learning 
rate does not strongly affect overfitting — its primary effect is on optimization stability and convergence speed. 3e-3 stands out as the clear optimum in this range.

**3.2 Effect of Dropout (Sweep 1)**
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/573fab92-4105-4225-be8f-c5521f02332e" />

This sweep varied dropout rate from 0.0 to 0.5. Both training and validation loss increase monotonically as dropout increases: training loss rises from ~1.56 
at dropout 0.0 to ~1.90 at dropout 0.5, and validation loss rises from ~1.65 to ~1.96 over the same range. Unlike the classic regularization story where dropout
narrows the train/validation gap at the cost of some training loss, here the gap between the two curves stays roughly constant (~0.08–0.10) across the entire sweep. 
This indicates dropout is not helping generalization in this setting — it is simply making the model harder to fit without a compensating reduction in overfitting. 
The most likely explanation is that the model is not overfitting much to begin with at this baseline configuration (given the relatively small gap even at dropout 0.0),
so dropout's regularizing effect has no overfitting problem to solve, and its cost — reduced effective capacity per training step — dominates.

**3.3 Effect of Number of Transformer Layers**
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/3be37a80-9cc1-47eb-ba7d-f8d7658b061c" />

Depth was swept from 2 to 10 layers. Training loss decreases fairly steadily with depth, from ~1.62 at 2 layers to ~1.54 at 8 layers, then plateaus (~1.55 at 10 layers).
Validation loss tells a more interesting story: it drops from ~1.70 (2 layers) to a minimum of ~1.61 at 8 layers, but then increases again to ~1.63 at 10 layers — 
while training loss keeps flat or slightly improves. This divergence at 10 layers, where validation loss increases even as training loss holds steady, is a signature of 
mild overfitting: the extra capacity from added depth is being used to fit training-set-specific patterns rather than generalizable structure. 8 layers appears to be the 
sweet spot for this configuration, balancing capacity against overfitting risk.

**3.4 Effect of Embedding Dimension**
![<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/b46c2d7c-fadd-4f8b-9590-403a5a410e19" />
]
Embedding dimension was tested at 32, 64, 128, and 256. Both losses drop sharply from 32 to 64 (training: ~1.68 → ~1.55; validation: ~1.75 → ~1.61), stay roughly flat 
from 64 to 128 (best point, training ~1.53, validation ~1.62), and then increase dramatically at 256, with both losses jumping to ~2.35–2.36 — worse than even the
smallest embedding size tested. This is a much larger and more abrupt degradation than the gradual overfitting seen with layer depth, and both training and validation 
loss rise together, which rules out ordinary overfitting as the explanation (overfitting would show training loss staying low while only validation loss rises). 
More likely causes are that the 256-dimensional model was undertrained relative to its larger parameter count within a fixed step budget, or that the increased embedding 
size required a corresponding learning rate adjustment that wasn't made, destabilizing training. This result underscores that embedding dimension can't be scaled up in 
isolation — it needs to be paired with adjustments elsewhere (learning rate, training steps, or both).

**3.5 Effect of Batch Size**
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/7c7fa7a1-2c7c-4907-87eb-00a5c6b0251c" />

Batch size was swept across 16, 32, 64, 128, 256, and 512. Both training and validation loss decrease monotonically and substantially across the entire range: training loss 
falls from ~1.55 at batch size 16 to ~1.25 at batch size 512, and validation loss falls from ~1.61 to ~1.40 over the same range. Notably, there is no sign of the 
improvement plateauing within the tested range — 512 is still the best setting. The train/validation gap narrows slightly at larger batch sizes, suggesting larger 
batches are not just accelerating optimization but also mildly improving generalization, likely via more stable, lower-variance gradient estimates. This is the strongest
and most consistent monotonic effect observed across all seven experiments, and it suggests batch size may be worth pushing even higher than 512 in follow-up work.

**3.6 Effect of Block Size (Context Length)**
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f0f4e9ef-f04e-4d22-991b-9ad695c16bff" />

Block size (the context window the model attends over) was tested at 32, 64, 128, and 256 tokens. Like batch size, this shows a clean monotonic improvement: training loss 
drops from ~1.38 at block size 32 to ~1.19 at block size 256, and validation loss drops from ~1.49 to ~1.33 over the same range. The rate of improvement is decreasing 
(the gain from 128→256 is smaller than from 32→64), suggesting diminishing returns are beginning to set in, but the curve has not yet flattened out entirely by 256. 
This makes intuitive sense: longer context gives the model more information to condition its predictions on, up to the point where most of the useful predictive signal 
in the data has already been captured.

**3.7 Effect of Dropout (Sweep 2)**
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/6a196137-ccb5-4d16-8b47-3cb06a09a4cd" />

A second, finer-grained dropout sweep was run over 0.0, 0.05, 0.1, 0.15, and 0.2, using a different baseline configuration than Sweep 1. The pattern is the same
direction as before: both training and validation loss increase monotonically as dropout increases. Training loss rises from ~1.78 at dropout 0.0 to ~2.02 at 
dropout 0.2, and validation loss rises from ~1.84 to ~2.06 over the same range. As in Sweep 1, the gap between training and validation loss stays roughly constant 
throughout, rather than narrowing. This second, independent sweep — despite different baseline settings — corroborates the finding from Sweep 1: in this modeling 
setup, dropout consistently hurts both training and validation performance without providing a generalization benefit, implying dropout is not addressing any 
overfitting problem for this model/dataset combination.

**4. Synthesis**
Bringing the seven experiments together, a few clear themes emerge:

1) ***Optimization-related hyperparameters (learning rate) show non-monotonic, U-shaped behavior. There is a clear optimum (3e-3 in this study), with both under- and 
over-shooting leading to worse performance. This is the classic signature of an optimization hyperparameter rather than a capacity or regularization one.***

2) ***Regularization (dropout) is actively hurting performance here. Both dropout sweeps agree: increasing dropout monotonically increases both training and 
validation loss without narrowing the generalization gap. This suggests the model, at its current scale and training budget, is not in an overfitting regime, 
and applying dropout removes useful capacity without solving a problem that doesn't yet exist.***

3) ***Capacity-related hyperparameters (layers, embedding dimension) show a "sweet spot then degradation" pattern, but for different underlying reasons. Layer count 
degrades gracefully and shows classic overfitting (validation loss rises while training loss holds steady) beyond 8 layers. Embedding dimension degrades catastrophically 
at 256, with both training and validation loss rising together — pointing to an optimization/training-budget failure rather than overfitting.***

4) ***Data-exposure hyperparameters (batch size, block size) show clean monotonic improvement across the entire tested range, with no evidence of a turning point yet. 
These are the two hyperparameters most likely to yield further gains if pushed beyond the ranges tested here.***

Overall, the best-performing configuration observed across all sweeps combines a learning rate near 3e-3, no dropout, roughly 8 transformer layers, 
an embedding dimension around 64–128, the largest tested batch size (512), and the largest tested block size (256).

**5. Limitations**

1)***Single-variable sweeps don't capture interactions.*** Each hyperparameter was varied independently while holding others fixed, so we cannot be certain the optimal value
found for one hyperparameter (e.g., 8 layers) remains optimal once other hyperparameters (e.g., batch size 512, or embedding dimension 128) are changed simultaneously.

2)***No error bars or repeated runs.*** Each configuration appears to reflect a single training run. Without repeated trials, we cannot distinguish genuine effects 
from run-to-run noise, particularly for the more abrupt changes observed (e.g., the embedding dimension 256 collapse).

3)***Two dropout sweeps used different baselines and ranges,*** which strengthens confidence in the qualitative direction of the effect but makes precise quantitative 
comparison between the two sweeps difficult.

4)***Untested regions beyond the sweep ranges.*** Batch size and block size both showed continued improvement at the largest tested value, meaning the true optimum for
these parameters is unknown and likely lies beyond the tested range.

5)***Possible confound in the embedding dimension result.*** The sharp degradation at 256 could stem from several distinct causes (insufficient training steps, need for 
learning rate rescaling, instability), and the current data can't distinguish between them.

6)***Fixed training budget.*** If total training steps or wall-clock time was held constant across all sweeps, larger-capacity configurations (more layers, larger embeddings) 
may be systematically disadvantaged relative to what they could achieve given proportionally more training.

**6. Future Work**

1) ***Joint/grid search over the most promising hyperparameters*** (learning rate, batch size, block size, layer count) to check whether their individually-optimal
   values remain optimal in combination.

2) ***Extend batch size and block size sweeps further*** (e.g., batch sizes beyond 512, block sizes beyond 256) to locate the point of diminishing or negative returns.

3) ***Diagnose the embedding dimension 256 failure directly*** by re-running that configuration with a lower learning rate and/or more training steps,
   to determine whether the issue is optimization instability or undertraining rather than a fundamental capacity problem.

4) ***Revisit dropout under conditions more likely to induce overfitting*** — e.g., a smaller dataset, more training epochs, or a larger model — to check whether dropout's
   benefit only appears once genuine overfitting is present.

5) ***Add repeated runs with different random seeds*** for at least the top few configurations to quantify run-to-run variance and confirm the robustness of the reported optimum.

6) ***Track loss curves over training, not just final values,*** to distinguish between configurations that are still improving when training stops versus those
   that have truly converged or begun overfitting.

**7. Conclusion**

This study systematically evaluated the impact of seven key hyperparameters on the performance of a Transformer-based language model. The experiments demonstrated that
hyperparameter selection plays a crucial role in achieving efficient optimization and strong generalization. Among the parameters investigated, the learning rate had 
the greatest effect on convergence, while larger batch sizes and block sizes consistently improved performance. An embedding dimension of 128 and eight Transformer 
layers provided the best trade-off between model capacity and generalization, whereas dropout was found to be unnecessary for the relatively small dataset.

The findings show that substantial performance improvements can be achieved through careful hyperparameter optimization alone, without altering the Transformer architecture.
These results also establish a strong baseline for future work involving larger datasets, modern Transformer variants, and interpretability techniques, making this study 
a practical foundation for more advanced language modeling research.
