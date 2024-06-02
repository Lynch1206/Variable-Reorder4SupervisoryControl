# Data embedding for supervisory control theory cases
> The purpose of the git repository is to demonstrate part of my work.
> The benchmark cases are sourced from the [ESCET software application](https://eclipse.dev/escet/).
> The supervisor synthesis algorithm that is used in this demonstration is data-synthesis.

Supervisory control theory is brought up when early 80s', and is driven ever since. It is actually a powerful idea that modeling, analysis and control a mega system that includes tons of individual self-motion components. <br/>

Simply put, the supervisory control theory is driven by discrete events, and it collects as much as information in the individual plants such that a supervisor can supervise all plants in the way as the engineer desires. <br/>

A system includes 2 major things: plants that describe all components' behaviors and requirements that describe what the system may do. <br/>

# A case example to demonstrate reordering
In the file of [Transitional_relationships.ipynb](./Transitional_relationships.ipynb), a soft introduction of transitional relationships is given. <br/>
In the file, there 2 cases. One is a simple case that I show how to derive the incidence matrix and adjacency matrix from the edges and the order of the nodes. <br/>
Further, I show a larger case in terms of number of nodes to demonstrate the effective of reordering according to the adjacency matrix. <br/>
<!-- show picture -->
![reordering](./Spectral_reorder.png)
