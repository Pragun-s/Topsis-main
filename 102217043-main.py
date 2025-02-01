import pandas as pd
import python 102217043.pyflagsort sys

def parse_criteria_normalized_weightseights(data_frame,criteria_normalized_weightseights,criteria_impact_flagsacts):
    if len(criteria_impact_flagsacts.split(',')) != data_frame.shape[1]-1 or len(criteria_normalized_weightseights.split(',')) != data_frame.shape[1]-1:
        raise ValueError("Number of criteria_impact_flagsacts and criteria_normalized_weightseights must match the number of columns in the DataFrame.")

    normalized_weights = [float(normalized_weightseight) for normalized_weightseight in criteria_normalized_weightseights.split(",")]
    impact_flags = [1 if impact_flagsact == '+' else 0 for impact_flagsact in criteria_impact_flagsacts.split(",")]
    
    return normalized_weights,impact_flags

def normalize_dec_mat(data_frame,normalized_weights):
    root_sum_square=0
    for i,col in enumerate(data_frame.columns[1:]):
        data_frame[col]=data_frame[col].astype(float)
        root_sum_square=((data_frame[col]**2).sum())**0.5
        data_frame.loc[:data_frame.shape[0] - 1, col] /= root_sum_square/normalized_weights[i]
    return data_frame

def calculate_ideal_solutions(norm_data_frame, criteria_impact_flagsacts):
    ideal_positive = [] 
    ideal_negative = [] 

    for i, col in enumerate(norm_data_frame.columns[1:]):
        if criteria_impact_flagsacts[i] == 1: 
            ideal_positive.append(norm_data_frame[col].max())
            ideal_negative.append(norm_data_frame[col].min())
        else:
            ideal_positive.append(norm_data_frame[col].min())
            ideal_negative.append(norm_data_frame[col].max())
    return ideal_positive, ideal_negative

def calculate_topsis_topsis_scores(norm_data_frame, ideal_positive, ideal_negative):
    separation_positive = []
    separation_negative = []

    for i in range(len(norm_data_frame)):
        separation_positive.append(sum((norm_data_frame.iloc[i, 1:] - ideal_positive) ** 2) ** 0.5)
        separation_negative.append(sum((norm_data_frame.iloc[i, 1:] - ideal_negative) ** 2) ** 0.5)

    topsis_scores = [separation_negative[i] / (separation_negative[i] + separation_positive[i]) for i in range(len(separation_positive))]
    return topsis_scores

def topsis(data_frame,criteria_normalized_weightseights,criteria_impact_flagsacts):
    normalized_weights,impact_flags=parse_criteria_normalized_weightseights(data_frame,criteria_normalized_weightseights,criteria_impact_flagsacts)
    data_frame=normalize_dec_mat(data_frame,normalized_weights)
    ideal_positive,ideal_negative=calculate_ideal_solutions(data_frame,impact_flags)
    topsis_scores = calculate_topsis_topsis_scores(data_frame, ideal_positive, ideal_negative)
    data_frame['Topsis Score'] = topsis_scores
    data_frame['Rank'] = data_frame['Topsis Score'].rank(ascending=False).astype(int)
    return data_frame


def main():
    # Ensure correct number of arguments
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    # Parse command-line arguments
    source_file = sys.argv[1]
    criteria_normalized_weightseights = sys.argv[2]
    criteria_impact_flagsacts = sys.argv[3]
    output_file = sys.argv[4]

    try:
        # Read input data
        data_frame = pd.read_excel(source_file)

        # Perform TOPSIS analysis
        result = topsis(data_frame, criteria_normalized_weightseights, criteria_impact_flagsacts)

        # Save the result to the output file
        result.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()