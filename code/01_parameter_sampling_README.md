# Cooling parameter sampling

Reads lower and upper bounds for each cooling technology, preserves the parameter order expected by the simulation functions, and generates Latin hypercube samples for 12 cases. Cases 11 and 12 use immersion cooling.

## Inputs

A Table B1 Excel workbook containing parameter bounds. The workbook path and sheet name are set in the notebook.

## Outputs

One case-numbered CSV per cooling case. Use the generated CSVs as inputs to the cooling simulation.
