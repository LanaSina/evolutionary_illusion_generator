# EIGen: Evolutionary Illusion Generator

## Acknowledgments

This tool relies on a cost function derived from a model built fron the architecture of W. Lotter and al. (https://coxlab.github.io/prednet/), model implemented, improved and trained by E. Watanabe (https://github.com/eijwat/Predictive_Coding_Deep_Neural_Networks, https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00345/full); on the pyTorch implementation of Ken Stanley's NEAT (https://github.com/uber-research/PyTorch-NEAT); and on useful blog posts by David Ha about graphics generation (https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/)

## Concept

How is prediction related to illusions? Watch the talk here: https://www.youtube.com/watch?v=jmC3B-tM2Zc

EIGen generates static images that appear to move: motion illusions. It is based on predictive coding:

![EIGen structure](_EIGen.png)

## Quick Start

Open illusion_generation.ipynb in Colab: https://colab.research.google.com/github/LanaSina/evolutionary_illusion_generator/blob/master/illusion_generation.ipynb

Run each cell as needed.

## Results

EIGen outputs both the current best pattern and the expected motion vectors.

![circles](gallery/centipede.png)
![vectors](gallery/centipede_vectors.png)

<img src="gallery/succulents.png"  width="250"> <img src="gallery/small_bw_1.png"  width="250"> <img src="gallery/small_bw_3.png"  width="250">

![big_color](gallery/color_big.png)
![big_bw](gallery/big_bw.png)

You can also choose a different structure(bands):

![bands](gallery/_bands.png)

The algorithm does not stop automatically. Stop it when you are happy with the results or when it's stagnating.

## I don't see the illusion!!! You're a hack!!!

Not everyone can see illusions. For an average motion illusion, ~25% of people might report seeing something.
Sometimes they are strong, sometimes they are weak. Sometimes the direction reverses depending if your lights are bright or if it's dark. Sometimes you get used to an illusion and it stops working.

The best way to see it is, in my experience:

	Be in a brightly lit environment, with the screen light on max

	Let your eyes wander on the screen or fixate a point in the distance

	Don't stare at the image directly!

Good luck!
