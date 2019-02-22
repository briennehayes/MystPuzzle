# MystPuzzle

Code for an in-progress research project, finding solutions to generalizations of a puzzle from Myst 4: Revelations

Current developments: successfully implemented a recursive search for finding solvable configurations of a graph, but recursion goes so deep in all but most basic graphs that exceptions occur. Working to implement a breadth-first search that will circumvent recursion depth limitations.

myst_classes.py
  - contains classes used for puzzle solving.
  
MystPuzzle.ipynb
  - primary notebook for documenting the project and exploring graph data. Contains information about the project and puzzles and reports present results. 

all_graphs.txt
  - contains descriptions, including edge lists, of all connected graphs with 4, 5, 6, 7, and 8 vertices. Graph information originally taken from Brendan McKay's combinatorial data webpage, http://users.cecs.anu.edu.au/~bdm/data/graphs.html.