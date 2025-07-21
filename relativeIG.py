import pandas as pd
import numpy as np
from typing import List
import io
from data import load_response

# adopt from faith and fate
def entropy(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return 0.0
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

def conditional_entropy(y_series: pd.Series, x_series: pd.Series) -> float:
    df = pd.DataFrame({'Y': y_series, 'X': x_series}).dropna()
    if df.empty:
        return 0.0
    total_entropy = 0.0
    x_probs = df['X'].value_counts(normalize=True)
    for x_val, p_x in x_probs.items():
        subset_y = df[df['X'] == x_val]['Y']
        h_y_given_x = entropy(subset_y)
        total_entropy += p_x * h_y_given_x
    return total_entropy

def relative_ig(y_series: pd.Series, x_series: pd.Series) -> float:
    h_y = entropy(y_series)
    if h_y == 0:
        return 0.0
    h_y_given_x = conditional_entropy(y_series, x_series)
    information_gain = h_y - h_y_given_x
    return information_gain / h_y

def calculate_positional_relative_ig_simple(responses, insertion_position=5):
    parsed_data = []
    max_sum_length = 0

    for response in responses:
        if '=' not in response:
            continue

        problem_part, sum_part = response.split('=')
        x_str, y_str = problem_part.split('+')

        parsed_data.append({
            'x_str': x_str,
            'y_str': y_str,
            'sum_str': sum_part
        })
        max_sum_length = max(max_sum_length, len(sum_part))

    results = []

    for k in range(max_sum_length):
        # Collect digits at position k
        x_k_list = []
        y_k_list = []
        z_k_list = []

        for data in parsed_data:
            x_k = data['x_str'][k] if k < len(data['x_str']) else 'PAD'
            y_k = data['y_str'][k] if k < len(data['y_str']) else 'PAD'
            z_k = data['sum_str'][k] if k < len(data['sum_str']) else 'PAD'

            x_k_list.append(x_k)
            y_k_list.append(y_k)
            z_k_list.append(z_k)

        # Create Series
        x_k_series = pd.Series(x_k_list)
        y_k_series = pd.Series(y_k_list)
        z_k_series = pd.Series(z_k_list)
        j = insertion_position

        # Calculate RelativeIG(z_k, {x_k, y_k})
        xy_combined = pd.Series([f"{x}|{y}" for x, y in zip(x_k_list, y_k_list)])
        rel_ig_k = relative_ig(z_k_series, xy_combined)
        
        # Calculate RelativeIG(z_{k-1}, {x_k, y_k})
        rel_ig_k_minus_1 = None
        if k > 0:
            z_k_minus_1_list = []
            for data in parsed_data:
                z_k_minus_1 = data['sum_str'][k-1] if (k-1) < len(data['sum_str']) else 'PAD'
                z_k_minus_1_list.append(z_k_minus_1)
            z_k_minus_1_series = pd.Series(z_k_minus_1_list)
            rel_ig_k_minus_1 = relative_ig(z_k_minus_1_series, xy_combined)
        
        # Calculate RelativeIG(z_{k+1}, {x_k, y_k})
        rel_ig_k_plus_1 = None
        if k + 1 < max_sum_length:
            z_k_plus_1_list = []
            for data in parsed_data:
                z_k_plus_1 = data['sum_str'][k+1] if (k+1) < len(data['sum_str']) else 'PAD'
                z_k_plus_1_list.append(z_k_plus_1)
            z_k_plus_1_series = pd.Series(z_k_plus_1_list)
            rel_ig_k_plus_1 = relative_ig(z_k_plus_1_series, xy_combined)

        if k < j:
            case_type = "k < j"
        else:
            case_type = "k >= j"

        results.append({
            'Position_k': k,
            'Case_Type': case_type,
            'RelativeIG_z_k': rel_ig_k,
            'RelativeIG_z_k_minus_1': rel_ig_k_minus_1,
            'RelativeIG_z_k_plus_1': rel_ig_k_plus_1
        })

    df = pd.DataFrame(results)

    return df

def analyze_responses(responses):

    df = calculate_positional_relative_ig_simple(responses, insertion_position=5)
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":

    response_10 = load_response('10+10_responses.txt')
    response_11 = load_response('11+10_responses.txt')
    response_12 = load_response('12+10_responses.txt')
    response_13 = load_response('13+10_responses.txt')
    results_10 = analyze_responses(response_10)
    results_11 = analyze_responses(response_11)
    results_12 = analyze_responses(response_12)
    results_13 = analyze_responses(response_13)