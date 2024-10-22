from typing import Dict, List,Tuple
import polyline
import math
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []  
    length = len(lst)
    
   
    for i in range(0, length, n):
        chunk = lst[i:i + n] 
       
        for j in range(len(chunk) // 2):
            chunk[j], chunk[len(chunk) - j - 1] = chunk[len(chunk) - j - 1], chunk[j]
        
        result.extend(chunk) 
    return result
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2)) 
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {} 
    for string in lst:
        length = len(string)  
        if length not in result:
            result[length] = []
        
        
        result[length].append(string)
    
    
    return dict(sorted(result.items()))

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"])) 
print(group_by_length(["one", "two", "three", "four"])) 


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
  
    def _flatten(current_dict: Any, parent_key: str = '') -> Dict:
        items = []
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key  
            if isinstance(value, dict): 
                
                items.extend(_flatten(value, new_key).items())
            elif isinstance(value, list):
                
                for index, item in enumerate(value):
                    list_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        items.extend(_flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, value))  
                
        return dict(items)

    return _flatten(nested_dict)


nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(path, used):
       
        if len(path) == len(nums):
            result.append(path[:])  
            return
        
        for i in range(len(nums)):
            
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            
            used[i] = True
            path.append(nums[i])
            
            
            backtrack(path, used)
            
            path.pop()
            used[i] = False
    
    nums.sort()
    
    result = [] 
    used = [False] * len(nums)  
    backtrack([], used)
    
    return result
print(unique_permutations([1, 1, 2]))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b'  
    date_pattern += r'|\b\d{2}/\d{2}/\d{4}\b'  
    date_pattern += r'|\b\d{4}\.\d{2}\.\d{2}\b'  

    matches = re.findall(date_pattern, text)
    
    return matches

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    R = 6371000  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
   
    coords = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
  
    df['distance'] = 0.0
    
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = polyline_to_dataframe(polyline_str)
print(df)



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    rotated_matrix = [[matrix[j][i] for j in range(n)] for i in range(n)]
    
    rotated_matrix = [row[::-1] for row in rotated_matrix]

    final_matrix = [[0] * n for _ in range(n)]  

    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix

matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
     df['timestamp'] = pd.to_datetime(df['timestamp'])
    results = []


    grouped = df.groupby(['id', 'id_2'])

    for (id_value, id_2_value), group in grouped:
        unique_days = group['timestamp'].dt.date.nunique()
        
   
        full_day_hours = len(pd.date_range(start=group['timestamp'].min().normalize(), 
                                            end=group['timestamp'].max().normalize(), 
                                            freq='H'))
        
   
        has_full_7_days = unique_days == 7
        has_full_24_hours = len(group['timestamp'].dt.hour.unique()) == 24

        results.append(((id_value, id_2_value), has_full_7_days and has_full_24_hours))

    result_series = pd.Series(dict(results))
    
    return result_series

df = pd.read_csv('dataset-1.csv')
boolean_series = time_check(df)
print(boolean_series)
