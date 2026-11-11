# Do AI-generated images help train better models than just rotating real ones?

Sarthak Agrawal | November 2026

We never have enough real microscope images to train a good predictor, so the usual move is to manufacture more training data out of the few we have. The cheap, standard way is rotation - spin and flip each real image - but a rotated colony is still the same colony. The alternative we're testing is generative AI: use an image model to invent new fake colonies that look real. The question is whether the AI route actually trains a better predictor than plain rotation when real data is scarce.

## How we tested it

Across four datasets, we trained the predictor five different ways and checked which did best on held-out real images it had never seen:

- real - just the handful of real images, no tricks.
- rotation - real images plus rotated and flipped copies. This is the baseline to beat.
- AI - real images plus AI-generated fakes, the thing under test.
- real-duplicated - real images plus exact copies of themselves. A control: if this matches rotation, the benefit was never "more images," it's variety.
- all-real - trained on all the real data, as a best-case ceiling.

Every test ran 15 times on fresh random splits and was scored on real images the model never trained on, so the numbers hold up.

## What the images look like

Each row below is one colony: first the real image, then that same colony rotated (that's all rotation does), then three AI-generated versions. They land close to real but not exact.

Branching colonies (bacteria growing in tree-like patterns):

![Branching: real, rotated, and AI-generated colonies](figs/samples_branching.png)

Mux patterns - here the colors carry the signal: red petals encode one chemical dial (aTc), green fill encodes another (IPTG), and the model has to read both back out.

![Mux: real, rotated, and AI-generated patterns](figs/samples_mux.png)

KL two-species mixing and Rattray strains, two more colony types:

![KL: real, rotated, and AI-generated](figs/samples_kl.png)

![Rattray: real, rotated, and AI-generated](figs/samples_rattray.png)

Worth flagging now: the AI branching colonies carry a faint color tint even though real colonies are basically gray. That detail comes back later.

## What the measurements mean

A few terms in the graphs are pipeline jargon, so here's a quick visual guide:

![Visual guide to the measurement terms](figs/metrics_explainer.png)

- shape_factor - round versus branchy. A compact blob scores high, a spread-out branchy one scores low.
- branch-architecture - how intricate the branching is, from counting the tips (red) and junctions (blue) in the colony's skeleton.
- aTc / IPTG - the two chemical dials the mux model reads back: aTc is the red petals, IPTG is the green fill.
- mixing ratio (kl) - how much of species A versus B.
- strain (rattray) - which of 62 variants.
- periodicity - how wavy the pattern's outline is.
- R-squared - a 0-to-1 score for predicting a number: 1 is perfect, 0 is no better than guessing the average, negative is worse.
- accuracy - fraction correct, for the pick-a-category tasks.

---

# The results

## The summary

![Summary of every comparison: green = AI beats rotation, red = AI loses](figs/fig1_headline_forest.png)

This figure stacks every comparison onto one chart. Each row is a task and the exact thing we measured; the dot shows how far the AI augmenter beat rotation (right, green) or lost to it (left, red), with the center line as a tie. If AI augmentation were broadly useful we'd see green all down the right side. Instead only two rows come out green and the rest are red, so it doesn't help in general - it wins on two specific measurements and loses everywhere else. The next charts break those down.

## Branching: AI helps the branch structure, not the overall shape

![Branching results: shape on the left, branch structure on the right](figs/fig2_branching_scarcity.png)

Both charts are branching colonies - the left predicts overall shape (round vs branchy), the right counts the branch tips and junctions. Moving right means more real training images, higher means more accurate, and each line is one recipe (red is AI, black is rotation). We hoped the red line would sit above black in both. It splits instead: on shape the AI line is near the bottom, worse than rotation, but on branch structure it's at the top, better. The likely reason is that the generator gets the branch skeleton right but draws slightly messy edges, and shape is measured from the edges while branch structure is measured from the skeleton.

### The color-tint check

![The color-tint test: orange is with tint, blue is after removing it](figs/fig4_tint_confound.png)

The AI fakes are faintly tinted while real colonies are gray, so before trusting the shape result we checked whether the model was just tripping on color. We redid the shape test with every image forced to grayscale (orange is the original, blue is de-tinted; below the line means AI lost). The blue bars are clearly less bad than the orange, so the tint really was hurting, but they stay below the line, so AI still loses on shape even once color is out of the picture. The tint was a real issue worth fixing, just not the whole story.

## Mux: the same fakes help one dial and hurt the other

![Mux results: reading the aTc dial (left) vs the IPTG dial (right)](figs/fig3_mux_axis_selectivity.png)

Both charts are the mux decode - reading aTc on the left and IPTG on the right out of the pattern, higher being more accurate, red the AI recipe and black rotation. On aTc the AI line rides above rotation, a real win; on IPTG it drops below. Same synthetic images, opposite outcomes, which means the generator captures the aTc signal (the red petals) faithfully but not the IPTG signal (the green fill). It's the clearest case in the study of the answer depending on exactly what you measure.

## The clear losses: mixing ratio, strain, waviness

![Three tasks where AI clearly lost; all-real data is highest](figs/fig5_negatives.png)

Three more tasks where AI lost outright: the kl mixing ratio, the rattray strain ID, and mux pattern waviness (periodicity). Each bar is a recipe's score, taller is better. The AI bars are among the shortest, rotation sits clearly higher, and all-real data is highest by a wide margin. On these the fakes just don't carry the right information, so adding them is worse than not bothering, and more real data is the only thing that moves the needle.

---

## The bottom line

Across four datasets, AI-generated images beat plain rotation in exactly two places - reading the aTc dial in mux, and the branch structure in branching colonies - and lost everywhere else. Three things make me trust that read: rotation is a genuinely strong baseline (a rotated image keeps its exact label, so it's clean free data), the duplication control roughly matched rotation (so the gains were about variety, not image count), and more real data beat every trick every time.

So in-domain AI augmentation turns out to be picky: it helps only when the fakes faithfully capture the specific thing you're measuring and that measurement tolerates their imperfections. The careful setup - clean dish-removed data, 15 repeats, the control arms, and catching the color-tint confound - is what lets us say that with confidence rather than guess. The open question is whether the two wins are worth pushing further, or whether the clean "it's axis-selective" story is the one to write up.
