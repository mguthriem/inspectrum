# Project Overview

This project is am R&D testbed for a planned utility to allow pre-processing of spectra prior to submission for analysis using Rietveld refinement. It will consist of a set of modules that will be tested with the hope that the modules that work well will become part of a deployed tool. This tool will assign good initial values to parameters such as lattice parameters, scale factor and peak widths needed for Rietveld analysis
 

## Project Name
<!-- What's your package called? -->
Inspectrum 

## Purpose
<!-- In 2-3 sentences, what problem does your package solve? -->
Certain Rietveld code can easily become unstable if initial values for its model parameters are not a close match to the experimental data. By pre-inspecting the spectrum to obtain good initial values, `inspectrum` will give good starting point for refinement. The information obtained from pre-inspection is expected to have ancillary usage for background subtraction. Importantly, `inspectrum` is designed from ground up to handle challenging data quality and will be tested against high-pressure neutron diffraction data, which features: high, structured background, low-signal-to-background and significant noise.


## Target Users
<!-- Who will use this? (e.g., biologists, data scientists, yourself) -->
Rietveld user with interest in "real world" data typically from in situ, in operando experiments with gnarly backgrounds and low signal-to-noise.

## Core Functionality
<!-- What are the 3-5 main things your package should do? -->

1. This package can ingest diffraction data and crystallographic data regarding scattering species
2. By inspecting the input data, optimised parameters of the crystallographic model (e.g. scale factor and lattice parameters) will be determined
3. It will also support pre-processing of the input data, such as background subtraction that can make use of the optimised parameters


## Input/Output
<!-- What kind of data/files does it work with? What does it produce? -->

**Input:** 

* A set of 1d powder-diffraction spectra, typically 1 to 10 spectra
* A list of crystallographic objects one for each phase contributing to scattering in the spectra
* A description of the instrument 

**Output:** 

* optimised parameters for each crystallographic object
* optimised parameters for the instrument description
* processed output 1d powder-diffraction spectra


## Example Usage
<!-- How would someone use your package? Show a simple example. -->

```python
# Example:
from inspectrum import loadDiffractionData

#load experimental diffraction data (N 1d spectrums)
data = loadDiffractionData(data = "data.gsa", instrument="myInstrument.insptm") #by default use current gsas format, specifying instrument used to measure data

#load crystallographic object from standard .cif format file. There are two so #create a list

tungsten = loadCrystalObject("tungsten.cif")
sample = loadCrystalObject("mySample.cif")

crystalObjects = [tungsten,sample]

#attempt to optimise parameters in crystalObjects (e.g. lattice params, relative scale)

result = inspect(data, crystalObjects, optimizeCrystal = True)

for crystal in result.crystalObjects:
    print(crystal)

#attempt to optimise parameters in crystalObjects and instrument (e.g. peak width, scale factor...)    
    
result = inspect(data, crystalObjects, optimizeCrystal = True, optimizeInstrument=True)

for crystal in result.crystalObjects:
    print(crystal)

for param in result.instrument.params:
    print(param)

```

## Dependencies
<!-- Any specific libraries you know you'll need? (e.g., pandas, numpy, matplotlib) -->
- This depends on GSAS-II installation (this is not a standard pixi instal
- Need a cif loader and crystalObject (mantid can do this, but probably a simpler dependency))


## Technical Notes
<!-- Any other requirements or constraints? -->


---

## Next Steps

Once you've filled this out, ask Copilot:

```
"I've described my project in docs/project.md. 
Please assess the current template and create an itemized plan to implement this project."
```

Copilot will read your project description and create a step-by-step plan!
