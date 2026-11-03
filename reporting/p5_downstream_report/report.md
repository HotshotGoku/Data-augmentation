# Do AI-generated images help train better models than just rotating real ones?

**Sarthak Agrawal | November 2026**

## The question, in plain terms

We almost never have enough real microscope images to train a good predictor. So we want a way to make *more* training data out of the few real images we've got. There are basically two ways to do that:

1. **Rotation** - take each real image and spin or flip it. It's free and everyone does it, but a rotated colony is still the same colony.
2. **AI generation** - use an AI image model to invent brand-new fake colonies that look like the real ones.

The whole study is one question: **does the AI way train a better predictor than the simple rotation way, back when real data is scarce?** If it does, that's huge, because almost every biology lab is short on labeled images.

## How we tested it

For four different datasets, we trained the predictor five ways and saw which one did best at predicting from brand-new real images it had never seen:

- **real** - just the few real images (no tricks).
- **rotation** - real images plus rotated/flipped copies (the baseline to beat).
- **AI (augmenter)** - real images plus AI-generated fakes (the thing we're testing).
- **real-duplicated** - real images plus exact copies of themselves. A sanity check: if this does as well as rotation, then "more images" alone was never the point, variety is.
- **all-real (ceiling)** - trained on *all* the real data, as a best-case to compare against.

We ran every test **15 times** and always tested on real images the model never trained on, so the numbers are solid and not a fluke.

## What the actual images look like

Before the results, here's what we're working with. In each row below: the first image is a **real** colony, the second is that same colony **rotated** (that's literally all rotation does), and the last three are **AI-generated** versions of it. Notice they look close to real but not perfect.

**Branching colonies** (bacteria growing in tree-like patterns):

![Branching: real, rotated, and AI-generated colonies](figs/samples_branching.png)

**Mux patterns** (the colors matter here: the red petal shapes encode one chemical dial called aTc, the green fill encodes another called IPTG, and the model has to read both back):

![Mux: real, rotated, and AI-generated patterns](figs/samples_mux.png)

**KL two-species mixing** and **Rattray strains** (two more colony types):

![KL: real, rotated, and AI-generated](figs/samples_kl.png)

![Rattray: real, rotated, and AI-generated](figs/samples_rattray.png)

One thing to notice: the AI-generated branching colonies have a faint **color tint**, even though real colonies are basically gray. That small detail turns out to matter later.

## What the measurements mean

A few of the words in the graphs are jargon from our pipeline, so here's a plain-language guide with pictures first:

![Visual guide to the measurement terms](figs/metrics_explainer.png)

- **shape_factor** - how round versus branchy a colony is. A round, compact blob scores high; a spread-out branchy one scores low. (First two images above.)
- **branch-architecture** - how intricate the branching is, measured by counting the **tips** (red dots) and **junctions** (blue dots) in the colony's skeleton. More tips and junctions means a more complex tree. (Third image.)
- **aTc** and **IPTG** - the two chemical "dials" the mux model has to read back out of a pattern. aTc shows up as the **red radial petals**, IPTG as the **green fill**. (Fourth image.)
- **mixing ratio** (kl task) - how much of species A versus species B is in the colony.
- **strain** (rattray task) - which of 62 bacterial variants the colony is.
- **periodicity** - how wavy or petal-like the pattern's outline is.
- **R-squared** - a 0-to-1 score for predicting a number: 1.0 is perfect, 0 is no better than guessing the average, and negative is worse than guessing.
- **accuracy** - the fraction the model gets exactly right, for the pick-a-category tasks.

---

# The results

## The summary (one chart that says it all)

![Summary of every comparison: green = AI beats rotation, red = AI loses](figs/fig1_headline_forest.png)

**Terms here:** each row is one task paired with the exact thing it measures (all defined in the guide just above).

**What the graph is:** every comparison in the whole study, stacked on one chart. Each row is one thing we tried to predict. The dot shows whether the AI images beat rotation (dot to the **right**, colored **green**) or lost to it (dot to the **left**, colored **red**). The line down the middle is "they tied."

**What we wanted:** all the dots green and to the right, meaning AI-generated data helps everywhere.

**What we got:** only **2 dots are green**. Everything else is red.

**What it means:** AI-generated images don't help in general. They only beat rotation on two specific measurements, and rotation wins all the rest. The next charts zoom into each one.

## Branching colonies: AI helps count branches, but not overall shape

![Branching results: shape on the left, branch-counting on the right](figs/fig2_branching_scarcity.png)

**Terms here:** *shape_factor* = how round vs branchy the colony is; *branch-architecture* = the number of tips and junctions in its skeleton (see the visual guide above).

**What the graph is:** two charts for the branching colonies. The left one is the model predicting the colony's overall **shape** (round vs branchy). The right one is the model **counting the branch tips and junctions**. Going right along the bottom means we gave the model more real images; higher up means more accurate. Each colored line is one of our five recipes (red is the AI one, black is rotation).

**What we wanted:** the red AI line sitting above the black rotation line in both charts.

**What we got:** opposite results in the two charts. On the **left (shape)**, the red AI line is near the **bottom**, so AI is worse than rotation. On the **right (branch-counting)**, the red AI line is at the **top**, so AI is better.

**What it means:** the AI fakes help for counting branches but hurt for overall shape. The likely reason is that the AI gets the branchy skeleton right but draws slightly messy edges, and "shape" is measured from the edges while "branch count" is measured from the skeleton.

### Was it just the color tint?

![The color-tint test: orange is with tint, blue is after removing it](figs/fig4_tint_confound.png)

**Terms here:** *shape_factor* = how round vs branchy the colony is (see the visual guide above).

**What the graph is:** a test of that color tint we noticed earlier. Since the AI fakes are faintly colored and real colonies are gray, maybe the model was getting confused by the *color*, not the shape. So we redid the shape test with every image turned gray. Orange bars are the original (tinted) result, blue bars are after removing the tint. Below the middle line means AI lost to rotation.

**What we wanted:** to see if the tint was the real culprit. If the blue bars jumped above the line, the tint was the whole problem.

**What we got:** the blue bars are clearly **less bad** than the orange ones, so the tint really was hurting by a real amount. But the blue bars are **still below the line**, so AI still loses at shape even after the fix.

**What it means:** the color tint was a genuine problem worth fixing (good to know for the future), but it wasn't the whole story. AI honestly isn't better than rotation at predicting shape.

## Mux patterns: the same fakes help one dial but not the other

![Mux results: reading the aTc dial (left) vs the IPTG dial (right)](figs/fig3_mux_axis_selectivity.png)

**Terms here:** *aTc* = the red-radial-petals dial; *IPTG* = the green-fill dial (see the visual guide above). The model reads both back out of the image.

**What the graph is:** two charts for the mux patterns. The model reads two chemical dials out of the image: **aTc** on the left, **IPTG** on the right. Higher means it reads the dial more accurately. Red is the AI recipe, black is rotation.

**What we wanted:** red above black in both charts.

**What we got:** a split. For **aTc (left)**, the red AI line is **above** rotation, a real win. For **IPTG (right)**, the red AI line is **below** rotation, a loss.

**What it means:** the exact same AI images help the model read one dial but not the other. The AI captures the aTc signal (those red petal shapes) faithfully, but not the IPTG signal (the green fill). It's the cleanest example of "it depends on exactly what you're measuring."

## The clear losses: mixing ratio, strain, and waviness

![Three tasks where AI clearly lost; green "all real data" towers over everything](figs/fig5_negatives.png)

**Terms here:** *mixing ratio* = how much of species A vs B; *strain* = which of 62 bacterial variants; *periodicity* = how wavy the pattern outline is (see the visual guide above).

**What the graph is:** three more tasks where AI lost cleanly: predicting a two-species **mixing ratio** (kl), identifying the **strain** (rattray), and measuring pattern **waviness** (periodicity). Each bar is one recipe's score; taller is better.

**What we wanted:** the red AI bars to be tall.

**What we got:** the red AI bars are among the **shortest**. Rotation is clearly higher, and "all real data" (the green bar) towers over everything.

**What it means:** for these tasks the AI fakes just don't carry the right information, so adding them is worse than doing nothing clever. The thing that actually helps is simply more real data.

---

## The bottom line

Across all four datasets, AI-generated images beat the simple rotation trick in only **two specific spots** (reading the aTc dial in mux, and counting branches in branching colonies), and lost everywhere else. A few things make me confident in that:

- **Rotation is a surprisingly strong baseline.** A rotated real image keeps its exact label, so it's clean, honest extra data, and that's hard to beat.
- **The duplication control did about as well as rotation**, which tells us any benefit was never just "more images," it was about genuine variety.
- **More real data beat every trick**, every time (the green ceiling bar).

So the honest, useful takeaway is that in-domain AI augmentation is **picky**: it only helps when the AI fakes faithfully capture the exact thing you're measuring, and when that measurement can tolerate the AI's small imperfections. The careful setup here (clean dish-removed data, 15 repeats, the control arms, and catching the color-tint issue) is exactly what lets us say that with confidence instead of guessing. I'd love to talk through whether the two wins are worth pushing on, or whether this clean "it's axis-selective" story is the one to write up.
