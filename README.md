# TOPSIS-Python

A Python package for implementing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method.

## Publishing

This package is published on [pypi.org](https://pypi.org/project/topsis-Shivansh-102203508/)

## Description

TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) is a multi-criteria decision analysis method. This implementation helps decision makers to find the best alternative from a finite set of alternatives based on multiple criteria.

The package provides functionality to:
- Normalize decision matrices
- Apply user-defined weights to criteria
- Handle both positive and negative impact criteria
- Calculate relative closeness to ideal solutions
- Rank alternatives based on TOPSIS scores

## Installation

You can install the package via pip:

```bash
pip install topsis-Shivansh-102203508
```

## Usage

### Command Line Interface

```bash
topsis <InputDataFile> <Weights> <Impacts> <ResultFileName>
```

### Parameters:

- `InputDataFile`: Excel file (.xlsx) containing the decision matrix
- `Weights`: Comma-separated weights for each criterion (e.g., "0.25,0.25,0.25,0.25")
- `Impacts`: Comma-separated impacts for each criterion ('+' for positive, '-' for negative) (e.g., "-,+,+,+")
- `ResultFileName`: Output Excel file name where results will be saved

### Example:

```bash
topsis input.xlsx "0.25,0.25,0.25,0.25" "+,+,-,+" result.xlsx
```

### Input File Format

Your input Excel file should be structured as follows:
- First column: Alternative names/identifiers
- Subsequent columns: Criteria values
- First row: Header with criteria names

Example:
```
Model   Price   Storage   Camera   Battery
A1      800     128      12       4000
A2      900     256      16       4500
A3      1000    512      20       5000
```

### Output

The program will generate an Excel file containing:
- All original columns
- Additional column 'Topsis Score'
- Additional column 'Rank'

## Error Handling

The package handles various errors including:
- Incorrect number of command-line arguments
- Invalid input file format
- Mismatched weights or impacts
- File I/O errors

## Dependencies

- pandas
- sys

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Shivansh Gupta

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Sample Code

```python
import pandas as pd
from topsis-Shivansh-102203508 import topsis

# Read your data
df = pd.read_excel('input.xlsx')

# Define weights and impacts
weights = "0.25,0.25,0.25,0.25"
impacts = "+,+,-,+"

# Calculate TOPSIS
result = topsis(df, weights, impacts)
```