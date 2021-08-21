# MOFI-Inversion

## Overview
This is the code bank used to generate the analysis in Patterson and Cardiff, (2021) *Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights*. At a high level this code generates synthetic data for single frequency or multi-frequency oscillatory flow interference tests using analytical solutions developed by Rasmussen et al., (2003). It then uses a gradient-based inversion strategy to quantify aquifer transmissivity, storativity, and leakance and their associated uncertainty.

The analysis presented in this paper was generated using code developed and run in MATLAB 2019b. This repository contains the MATLAB with plans to incrementally translate translate to Python as time and my dissertation allows.

## MATLAB_Code
* Confined_Analysis.m runs the analysis seen in the section **Confined Aquifer System** and generates Figures 2-5 found in this section.
* Leaky_SingleFreq.m runs the single frequency analysis seen in the section **Leaky Aquifer System** and generates Figure 6 found in this section.
* Leaky_MultiFreq.m runs the multi-frequency analysis seen in the section **Leaky Aquifer System** and generates Figures 7-9 found in this section.
* All .m files found in the Func_Lib subdirectory are dependencies that are called in the main codes mentioned above. Documentation and variable description is found at the top of each individual file.

## Python_Code
This section is currently less than barebones. I am incrementally translating the contents of the MATLAB Func_Lib subdirectory here as time allows.

## References
Patterson, J. R., M. A. Cardiff. 2021. Aquifer Characterization and Uncertainty in Multi-Frequency Oscillatory Flow Tests: Approach and Insights. Groundwater, Under Review

Rasmussen, T. C., K. G. Haborak, and M. H. Young. 2003. Estimating aquifer hydraulic properties using sinusoidal pumping at the Savannah River site, South Carolina, USA. Hydrogeology Journal 11, no. 4: 466–82, https://doi.org/10.1007/s10040-003-0255-7
