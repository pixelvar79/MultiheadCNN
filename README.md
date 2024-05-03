<h1 style="text-align: center;">Determining critical traits in bienergy crops using aerial imagery and a multitrait learning strategy</h1> 

-------
-------

Perennial crops like Miscanthus are highly productive plants with reduced carbon footprint, making them a promising alternative for sustainable provision of biofuels.

Large-scale plant breeding in crops like Miscanthus faces challenges such as long generation times and the need for manual and error-prone field measurements of thousands of plants for relevant phenotypic traits. 

A sequence of images are used as inputs into a spatiotemporal 3D- Convolutional Neural Network (3D-CNN) architecture [1] to determine key-traits such as culm length, culm basal outer diameter, and biomass in Miscanthus. 

A Multitask Learning (MTL) strategy was integrated into the 3D-CNN as efficient method of extracting multiple phenotypic traits by simultaneously learning them from the same input imagery which can contribute to improve the predictive ability and reduce manual phenotyping in the field.

To test our hypothesis, this work is organized as follows:
A single-trait (ST) 3D-CNN was implemented to predict each trait (culm length, culm basal outer diameter, biomass) of interest.
A multi-trait (MT) 3D-CNN was implemented to simultaneously predict each of the three traits.
The performance of each approach was evaluated in unseen (i.e., testing) data via MAE, MAPE, R2 metrics.



-------
-------

<p align="center">
  <img src="Screenshot1.png">
</p>


