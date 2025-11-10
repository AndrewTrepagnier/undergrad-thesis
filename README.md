# My research explained!

## Machine-Learned Interatomic Potential of Pure Crystalline Yttrium in Hexagonal Close-Packed form

I am the first author of the following material; however, this work would not have been possible without Dr. Christopher Barrett and Dr. Doyl Dickel. This research builds off of their work [here](https://www.sciencedirect.com/science/article/abs/pii/S0927025621002068)





![12377606-8449-4CED-B1C0-DEEC3649A4B4_1_105_c](https://github.com/user-attachments/assets/ea4ca987-ec34-4cb6-a4ea-749eeb7b2954)





### Abstract
The use of neural network architectures for interatomic potential fitting has recently become a computation
strategy of great interest in material science research due to
its superior ability handling large, memory-intensive datasets in
sufficient time. One challenge in modern molecular dynamics
is the inflexibility of conventional Modified Embedded Atom
Methods(MEAM) and Embedded Atom Method (EAM) many-body potentials due to their limiting number of characterizing
parameters. In this paper, we sought to approximate pure
Yttrium potentials using a feed-forward multi-layer perceptron
(MLP) neural network on Density Functional Theory(DFT)
simulated databases. For each potential(Energy-Volume Relation,
Elastic Constant, and Heat to Melt), our model demonstrated
significantly higher agreement with DFT simulated potentials and
empirical databases of pure Y than traditional MEAM and EAM
formalisms.

### Introduction

Pure-Crystalline Yttrium, or more commonly referred to as
”pure Y” or ”Y”, has had a growing body of research literature
due to its novelty enhancing strength-to-weight properties of
popular Mg-Al ternaries and binaries[]. Namely, MG-3 wt%Y
has been proven to have improved strength and corrosion
properties with the solid-solution hardening of pure Y, making
MG-Y systems of varying Y weight percentages a material of
interest for applications demanding high strength, lightweight
applications. More specifically, MG-Y systems have been used
in artificial knee joints, aircraft components, and various high-
performance automotive components. Thus, it is important that
highly accurate, well-defined energy potentials of pure Y are
published to support the development of more robust, complex
molecular alloys for more conditional applications.

The ever-growing study of elements like pure Y on the
structural level has introduced new challenges in material
science. The study of larger structures has inherently required
processing of larger volumes of data. On the small scale( for
lattice structures less than a few hundred atoms), electronic
structure calculations such as Density Functional Theory(DFT)
and first principle energy have proven to produce high quality,
accurate energy potentials when compared to published empirical databases. However, the performance of DFT simulations
taper off as larger structures consisting of several hundred
atoms are modelled.

To deal with this issue, Many-body interatomic potential calculations such as Modified Embedded Atom Methods (MEAM) and Embedded Atom Methods (EAM) formalisms
have traditionally been used to scale electronic structure data to
molecular dynamics and molecular statics simulations. Despite
EAM and MEAM formalisms semi-empirical premise, these
existing formalisms do not represent the breadth or complexity
of the interatomic energy potentials. Subsequently, MEAM
and EAM potentials produce inflexible inputs for molecular
dynamics simulations due to a limiting number of 13 free
parameters for characterization of the structure.

Given the need for a more accurate energy potential, we
sought to evaluate a machine learning framework to approximate metaparameters of pure Y’s elastic constants and energy-
volume relation. Machine learning is an excellent solution
for big data processing, as a well-architected framework can
interpolate complex data trends with thousands of parameters.
Our model uses energy and force DFT databases to construct
a high-resolution energy potential. Prior research literature in
machine learning fitted potentials have not only demonstrated
ML-enabled potentials to fit normal material behavior metrics
(i.e. elastic constants, energy-volume curves), but also shed
light onto more subtle behaviors such as temperature at phase
transformation or atomic structure during dislocation com-
pared to MEAM or EAM potentials. In this paper, we test and
validate the fidelity of a pre-trained machine learning model
for fitting DFT datasets for MD simulation in Large-scale
Atomic/Molecular Massively Parallel Simulator(LAMMPS).


### Methodology - Artificial Neural Network

Machine learning techniques have propelled many fields
to extract information from large volumes of data. Of these
techniques, feed-forward multi-layer perceptron (MLP) neural
networks have demonstrated impressive benchmarks of computing speed and accuracy. MLP architectures are biologically
inspired frameworks. The perceptrons of the model are mathematical functions that simulate neurons, the most fundamental
cell responsible for “thinking” in the human brain.

Previous approaches using neural networks for potentials,
such as the widely used Behler-Parrinello atom-centered symmetry function, have been demonstrated as an effective fingerprint for an artificial neural network (ANN) architecture by
several authors. Artrith and Urban approximated the potential of titanium oxide using the Behler-Parrinello basis function for
an ANN consisting of 2 hidden layers and 10 neurons each
[]. Additionally, the Behler-Parrinello basis functions were
utilized in Singraber, Behler, and Dellago authored n2p2 neural
network package which has been used to publish various
potentials of Al-Cu and Mg.

In the instance of pure Y, careful design must be considered
to make the structural fingerprint concise. Our model architecture uses an MLP rapid artificial neural network (RANN)
with embedded structural fingerprints based on the MEAM-formalism in the input layer. The MEAM formalism is the
best current method of characterizing the atomic environment
as it makes use of the angular screening between neighboring
atoms. Therefore, we evaluated a model that exploits the
underlying structure of data using an amenable number of
parameters for faster computation speeds.

### Methodology Cont. - Fingerprinting
The role of the neural network is to predict the energy of each atom given its respective local environment; therefore, two MEAM-based structural fingerprints are used. The foundation of the fingerprints consists of the vectors of radii between neighboring atoms in a 3-D space. The function of the pair interaction fingerprint, \(F_{n}\), is described as:

$$
F_n = \sum_{j \neq i} \left( \frac{r_{ij}}{r_e} \right)^n e^{-\alpha_n \frac{r_{ij}}{r_e}} f_c \left( \frac{r_c - r_{ij}}{\Delta r} \right) S_{ij}
$$

Where \(\alpha_{n}\) is based on the bulk modulus defined in MEAM. The neighbor cutoff distance, \(r_{c}\), and equilibrium nearest neighbor distance, \(r_{e}\), are displacements affecting the energy of the atom \(i\) and atom \(j\). The 3-body fingerprint, \(G_{m,k}\), is described below by:

$$
\begin{aligned}
G_{m,k} = \sum_{j,k} \cos^m \theta_{jik} 
    & \ e^{-\beta_k \frac{r_{ij} + r_{ik}}{r_e}} 
    f_c \left( \frac{r_c - r_{ij}}{\Delta r} \right) \\
    & \times f_c \left( \frac{r_c - r_{ik}}{\Delta r} \right) 
    S_{ij} S_{ik}
\end{aligned}
$$

Where \(\theta_{i,j,k}\) describes the interatomic angle between atom \(i\), atom \(j\), and atom \(k\). The pair interaction and 3-body fingerprints are collected as a vector that will serve as the input layer of the first neuron.

### Elastic Constants

In this section, we present numerical results of the elastic constants obtained using the RANN architecture. Due to the hexagonal lattice structure of pure Y, the selected tensors, \(C_{11}\), \(C_{12}\), \(C_{13}\), \(C_{33}\), \(C_{44}\) were used to capture its full stress and strain behaviors. In particular, \(C_{33}\) is a crucial tensor for understanding the material behaviors in the hexagonal axis of an anisotropic system. 

| Property | RANN (this work) | Exp.          | 2NN MEAM (ref.) | MEAM (ref.) |
|----------|------------------|---------------|-----------------|-------------|
| \(C_{11}\) | 80.76           | 83.40\(^{a}\) | 77.74\(^{b}\)   | 86.27\(^{c}\) |
| \(C_{12}\) | 20.52           | 29.10\(^{a}\) | 30.00\(^{b}\)   | 25.73\(^{c}\) |
| \(C_{13}\) | 21.74           | 19.00\(^{a}\) | 28.21\(^{b}\)   | 19.41\(^{c}\) |
| \(C_{33}\) | 80.59           | 80.10\(^{a}\) | 75.32\(^{b}\)   | 69.51\(^{c}\) |
| \(C_{44}\) | 25.96           | 26.90\(^{a}\) | 21.71\(^{b}\)   | 24.10\(^{c}\) |



References:

Dickel, D. E., Baskes, M. I., Aslam, I., & Barrett, C. D. (2018). New interatomic potential for Mg–Al–Zn alloys with specific application to dilute Mg-based alloys. Modelling and Simulation in Materials Science and Engineering, 26(4), 045010

Dickel, D., Barrett, C. D., Carino, R. L., Baskes, M. I., & Horstemeyer, M. F. (2018). Mechanical instabilities in the modeling of phase transitions of titanium. Modelling and Simulation in Materials Science and Engineering, 26(6), 065002

Dickel, D., Nitol, M., & Barrett, C. D. (2021). LAMMPS implementation of rapid artificial neural network derived interatomic potentials. Computational Materials Science, 196, 110481

