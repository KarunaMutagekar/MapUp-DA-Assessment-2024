import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    df = pd.read_csv('C:/Users/DELL/Desktop/Exam/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')

   
    toll_ids = pd.unique(df[['id_start', 'id_end']].values.ravel())
    toll_ids.sort()

    distance_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)
    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']

        distance_matrix.at[start_id, end_id] = distance
        distance_matrix.at[end_id, start_id] = distance

   
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

print(calculate_distance_matrix(df))


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


distance_matrix = calculate_distance_matrix('path_to_your_dataset.csv')
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
   distances = df[df['id_start'] == reference_id]['distance']
    
   
    if distances.empty:
        return []

    
    average_distance = distances.mean()

    
    lower_bound = average_distance * 0.90  
    upper_bound = average_distance * 1.10 

   
    filtered_ids = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]['id_start']

    
    unique_filtered_ids = sorted(filtered_ids.unique().tolist())

    return unique_filtered_ids

unrolled_df = unroll_distance_matrix(distance_matrix)
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001400)
print(result_ids)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
   rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df

unrolled_df = unroll_distance_matrix(distance_matrix)
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

   
    new_rows = []


    time_intervals = [
        (time(0, 0), time(10, 0), 0.8),   
        (time(10, 0), time(18, 0), 1.2),  
        (time(18, 0), time(23, 59, 59), 0.8),  
    ]
    
   
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distances = row[['moto', 'car', 'rv', 'bus', 'truck']]
        

        for day in days_of_week:
            for interval in time_intervals:
                start_time, end_time, discount_factor = interval

             
                if day in ['Saturday', 'Sunday']:
                    discount_factor = 0.7 

              
                new_row = {
                    'id_start': id_start,  
                    'id_end': id_end,      
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                }

                
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    new_row[vehicle_type] = distances[vehicle_type] * discount_factor

                new_rows.append(new_row)

   
    expanded_df = pd.DataFrame(new_rows)

    return expanded_df

unrolled_df = unroll_distance_matrix(distance_matrix)
toll_rate_df = calculate_toll_rate(unrolled_df)
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_df)