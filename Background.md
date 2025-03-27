# **Background & Related Work**

To explore the transformation of meaning across representational layers, we draw upon multiple intersecting fields—each offering pieces of the larger puzzle: how information is stored, compressed, expressed, and ultimately understood. This section outlines the core areas that inform our work.

---

### **2.1 Representation Learning**

At the foundation of this work is the principle of **representation learning**—the idea that neural networks can automatically discover useful features and structures in data without hand-designed encodings. In particular, autoencoders and their probabilistic cousin, **Variational Autoencoders (VAEs)** [Kingma & Welling, 2013], enable models to learn compressed, structured latent spaces that preserve salient characteristics of the input.

These learned spaces are more than technical artifacts—they are the **geometry of abstraction**. Each dimension, each cluster, captures something about the **essence** of the input. In our system, the latent space acts as a kind of *cognitive cortex*, holding the agent state in a transformed but faithful form.

Relevant work includes:
- Bengio et al. [2013] on the **Manifold Hypothesis**, which posits that high-dimensional data lies on lower-dimensional manifolds—ideal for compression and structure extraction.
- Oord et al. [2017] with **VQ-VAE**, introducing vector quantization in latent space to bridge discrete and continuous representations.

---

### **2.2 The Information Bottleneck Principle**

Our work is also grounded in the **Information Bottleneck (IB)** framework [Tishby et al., 2000], which formalizes the trade-off between compression and preservation of relevant information. The IB principle views learning as a balance: minimize mutual information with the input (compress), while maximizing mutual information with the target (relevance).

In our case, the target is not a classification label, but the **semantic integrity** of the original state. We ask not just whether the reconstructed state resembles the input, but whether it still *means the same thing*—structurally, functionally, and behaviorally.

This generalizes the IB principle toward **semantic bottlenecks**, where information is filtered not just for relevance in prediction, but for **coherence in transformation**.

---

### **2.3 Neural Compression**

Compression is traditionally seen as an algorithmic endeavor—zlib, LZ77, Huffman coding, etc.—but recent advances in **neural compression** have shown that learned models can outperform hand-crafted algorithms on domain-specific tasks.

Work by Balle et al. [2018] introduced **entropy bottlenecks**—learned distributions over latent variables that enable variable-length coding using arithmetic compression. These models blend VAEs with information theory, and their goal is often to reduce **bits per pixel** or **bits per symbol**.

While we draw inspiration from this, our goal differs. We are not optimizing for pixel-perfect fidelity or maximum compression ratio. Rather, we are investigating whether such compression can be **guided by a sense of meaning**—whether compression can serve understanding, not just storage.

---

### **2.4 Symbolic to Subsymbolic Translation**

Agent states, particularly those used in simulations or cognitive models, are often **symbolic**: dictionaries of key-value pairs, Boolean flags, structured knowledge. Neural networks, in contrast, operate in the **subsymbolic** realm—where meaning is distributed, implicit, and geometric.

Bridging these two domains is an open challenge in artificial intelligence. Our work adds to this effort by attempting to **translate symbolic state representations into latent codes** and back, with minimal semantic loss. This resembles work in:
- **Neuro-symbolic systems** [Garcez et al.]
- **Program synthesis from latent space**
- **Concept bottlenecks** [Koh et al., 2020], where latent spaces are supervised with semantic labels

But instead of explicitly mapping to concepts, we allow the system to learn what *meaning looks like*—implicitly and reversibly.

---

### **2.5 Philosophical and Cognitive Inspiration**

Beyond technical lineage, this work is motivated by deeper questions about **how meaning moves through form**. In cognitive science and philosophy of language, there is a long tradition of exploring how thought is expressed through symbol, speech, or gesture—each a transformation of the same internal essence.

- Ludwig Wittgenstein asked: *“What can be shown cannot be said.”*
- Gregory Bateson described information as *“a difference that makes a difference.”*
- Marvin Minsky viewed intelligence as the ability to shift perspectives between representations.

In a way, our model is not just a compression system—it is an attempt to model **perspective-shifting**: to see an agent state not just as data, but as something that can be *re-seen* through many layers and still retain what makes it *it*.

---

### **Summary**

This project stands at the intersection of representation learning, information theory, compression, and the study of meaning. By combining tools from these disciplines with the metaphorical lens of algorithmic anatomy, we aim to contribute not just a system, but a way of thinking about transformation, fidelity, and the preservation of essence across change.