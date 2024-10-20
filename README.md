# Row Crop Photosynthesis Model 
A mathematical model for calculating the canopy photosynthetic rate (A<sub>c</sub>) of a row-planted crop canopy. 

## Description
A<sub>c</sub> of a row-planted crop canopy can be calculated by numerically integrating the net leaf photosynthetic rate (A<sub>L</sub>) at grid points within the canopy. Here, A<sub>L</sub> is calculated by solving the simultaneous equations of the biochemical leaf photosynthesis model (<a href="https://link.springer.com/article/10.1007/BF00386231">Farquhar et al., 1980</a>), stomatal conductance model (<a href="https://link.springer.com/chapter/10.1007/978-94-017-0519-6_48">Ball et al., 1987</a>), and one-dimensional mass transport model (<a href="https://research.wur.nl/en/publications/photosynthesis-of-crop-plants-as-influenced-by-light-carbon-dioxi">Gaastra, 1959</a>) with the input of the absorbed PAR (I<sub>L</sub>) and other climatic variables. I<sub>L</sub> was calculated using the row-planted canopy model of <a href="https://www.sciencedirect.com/science/article/abs/pii/016819238990004X">Gijzen and Goudriaan (1989)</a>. 

For more details, please read our paper (under review).

## Demo
### 3D simulation
This animation was created by connecting output images. 
<img src="https://github.com/user-attachments/assets/ec06bd77-c015-4f4f-a84f-72cb941d8f35" width="60%"/>

### Cross-section
A snapshot of the cross-sectional distribution of A<sub>L</sub>.
![image](https://github.com/user-attachments/assets/059166e4-1a8b-4e96-a2e2-cd455f748cc3)
## Requirement
- The calculation is CPU-intensive because the model calculates A<sub>L</sub> at all the defined grid points in the 3D canopy (e.g., 100 x 100 x 100 = 10<sup>6</sup>  points). Therefore, we recommend using a good CPU (e.g., Intel Core i9) to run the calculation.
- Install the necessary packages in "requirements.txt".
- The codes were written in a Linux environment. You might encounter unexpected errors if you run the codes in Windows/Mac environments.
- We used Python version 3.10.12.

## Usage
- Parameters are stored in "./sample/parameter_list_v2.yml".
- Climate data are stored in "./sample/climate_data.csv".
- The time-series data of the canopy height (Hr) and leaf area index (LAI) are stored in "./sample/canopy_geometry.csv". This data should be further processed using "canopy_geometry.py" to create "canopy_geometry_processed.py".
- After the preparation of the above three files (i.e., parameter_list_v2.yml, canopy_geometry_processed.csv, climate_data.csv), you can run "canopy_photo_greenhouse_v2.py", and you will find output files in "./sample/output" (output directory will be automatically created). It will take several minutes, depending on your CPU.
- You can visualize the output file (in .feather extension) using "results_visualizer.py".
 
## Install
Unfortunately, we have not organized the necessary files and codes in a user-friendly way yet. Please just clone the repository, install the necessary packages written in "requirement.txt", and run. 

## Licence
[MIT](https://mit-license.org/)

## Author
[Koichi Nomura](https://github.com/nomurako)
