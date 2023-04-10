# HPC for Building Simulation

This repository has some examples for implementing numerical methods in the context of building simulation while leveraging high-performance computing resources (namely, GPUs).

It is very much in WIP state. 

## Heat Diffusion through Solids

Currently working on accelerated Finite Difference Method (FDM) solvers for unsteady heat diffusion through spatially varying diffusivity fields.  

The main use-case here is for 2D analysis of thermal bridges and breaks in building details/wall constructions.  

## Atria & Solar Chimney Natural Ventilation

Currently working on accelerated Lattice Boltzmann Method (LBM) solvers which couple thermal fields with hydrodynamic fields for the purpose of modeling buoyancy driven natural ventilation in buildings, i.e. for solar chimneys and atria, or data centers on high floors dumping heat into a ventilation shaft.

## Conjugate Heat Transfer

(NOT YET IMPLEMENTED)

## Integration with CAD

(NOT YET IMPLEMENTED) connection to Rhino/Grasshopper through websockets or HTTP requests
