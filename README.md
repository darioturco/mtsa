# MTSA #

*Modal Transition System Analyser*

[MTSA webpage](http://mtsa.dc.uba.ar/)

## What is MTSA? ##
The Modal Transition Analyser (MTSA) supports behaviour modelling, analysis and synthesis of concurrent software systems. 
Behaviour models are described using the Finite State Processes (FSP) language and behaviour constraints described in Fluent Linear Temporal Logic (FLTL). Descriptions are then be compiled into Labelled Transition Systems (LTS). 
Analysis is supported by model animation (LTS walkthroughs and graphical animation), FLTL model checking, simulation and bisimulation. 
Synthesis is supported by implementation of various discrete event controller synthesis algorithms.
MTSA also supports modelling, analysis and synthesis partial behaviour models in the form of Modal Transition Systems (MTS). 

MTSA is an evolution of the LTSA tool developed at Imperial College and is a currently a joint research effort of the Distributed Software Engineering (DSE) group at Imperial College London and the Laboratory on Foundations and Tools for Software Engineering (LaFHIS) at University of Buenos Aires. 

This version of MTSA is developer to be used for GR(1) problems. In this proyect Blockin is equibalent of GR(1) but is not a standar notation. The Non-Blocking code is not yet ready to be fully used.

## How to install ##
You can run MTSA using JAVA. Just download the [JAR]() and run it. Also you can find the java source code ready for compile in the folder mtsa of this repo.

## MTSAPy ##

This project extend the original MTSA proyect adding a python module that can use the DCSForPython interfaze. That allow the python script to use the DCS-OTF Blocking implementation of MTSA. This allow us to train, select, and test the RL agents.

## Projects ##

MTSA is an experimental platform for research in Software Engineering. 
Some research projects, with associated papers, case studies, and experimental data can be found at our [main webpage](http://mtsa.dc.uba.ar/#projects).

## Publications ##

Refer to our [main webpage](http://mtsa.dc.uba.ar/#publications) for more information on publications with MTSA.