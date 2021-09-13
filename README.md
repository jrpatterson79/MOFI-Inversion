# MOFI-Inversion

## Overview
This code in this repository is used to generate the analysis in Patterson and Cardiff, (2021) *Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights*. At a high level this code generates synthetic data for single frequency or multi-frequency oscillatory flow interference tests using analytical solutions developed by Rasmussen et al., (2003). It then uses a gradient-based inversion strategy to quantify aquifer transmissivity, storativity, and leakance and their associated uncertainty.

The analysis presented in this paper was generated using code developed and run in MATLAB 2019b. This repository contains the MATLAB with plans to incrementally translate translate to Python as time and my dissertation allows.

The code is provided as open source under the GNU General Public License v3.0. It is provided without warranty, but should perform as described in the manuscript when executed without modification. If you would like to use any of the code in this repository for research, software, or publications, we ask that you provide a citation to the code and journal article (See references below). 

## MATLAB_Code
* Confined_Analysis.m runs the analysis seen in the section **Confined Aquifer System** and generates Figures 2-5 found in this section.
* Leaky_SingleFreq.m runs the single frequency analysis seen in the section **Leaky Aquifer System** and generates Figure 6 found in this section.
* Leaky_MultiFreq.m runs the multi-frequency analysis seen in the section **Leaky Aquifer System** and generates Figures 7-9 found in this section.
* All .m files found in the Func_Lib subdirectory are dependencies that are called in the main codes mentioned above. Documentation and variable description is found at the top of each individual file.

## Python_Code
This section is currently less than barebones. I am incrementally translating the contents of the MATLAB Func_Lib subdirectory here as time allows.

## References
Patterson, J. R., M. A. Cardiff. 2021. Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights. Groundwater, https://doi.org/10.1111/gwat.13134

Rasmussen, T. C., K. G. Haborak, and M. H. Young. 2003. Estimating aquifer hydraulic properties using sinusoidal pumping at the Savannah River site, South Carolina, USA. Hydrogeology Journal 11, no. 4: 466–82, https://doi.org/10.1007/s10040-003-0255-7
