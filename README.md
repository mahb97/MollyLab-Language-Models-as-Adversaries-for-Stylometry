# MollyLab-Language-Models-as-Adversaries-for-Stylometry

μp → UP 
---

### μp → UP

Late-night thought: Joyce's "up:UP" from the Circe episode can be read as a formula:
- Let μ be the morpheme, the micro-unit.
- The arrow is the transformation.
- UP is the emergent voice.

**μp → UP** — *from the smallest unit of meaning, the whole voice rises.*

It maps onto the three-phase pipeline used in Wake2vec which was essentially the food for thought for this project:

- **P1** is learning the μ where raw embeddings for Joyce's morpheme splinters. Teaching the model that `funferall` and `chaosmos` and `cropse` exist as atomic units.
- **P2** is the arrow: LoRA training the attention layers to *route* those micro-units through the model's machinery, which is the transformation itself.
- **P3** enforces that the arrow preserves structure: direction consistency loss ensuring morpheme groups compose coherently. Not a lookup table but a proper morphism.

Other resonances worth noting for Wake2vec:
- μ in physics = coefficient of friction. The val plateau since step 1400 (Llama p1) *is* that friction: so the resistance between memorizing Wake tokens and learning to compose with them. P2/P3 are designed to overcome it.
- μ is Greek which is carrying the Homeric undertow that Ulysses is built on. The micro and the mythic sharing a letter.
- The colon in "up:UP" reads as "such that" or "yields" which can be a mapping, a type annotation, a definition, all of those work.

For the future paper, that formula anchors the methods section. 
