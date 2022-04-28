import pandas as pd
import numpy as np
import random
import logging as logger

logger.basicConfig(filename='private_histo.log', level=logger.INFO, force=True, filemode='a')

# Rename columns following the guidelines
col_dict = {'PUMA': 'PUMA_x',
            'YEAR': 'YEAR_x',
            'HHWT': 'HHWT_x',
            'GQ': 'GQ_c',
            'SEX': 'SEX_c',
            'AGE': 'AGE_c',
            'MARST': 'MARST_c',
            'RACE': 'RACE_c',
            'HISPAN': 'HISPAN_c',
            'CITIZEN': 'CITIZEN_c',
            'SPEAKENG': 'SPEAKENG_c',
            'HCOVANY': 'HCOVANY_c',
            'HCOVPRIV': 'HCOVPRIV_c',
            'HINSEMP': 'HINSEMP_c',
            'HINSCAID': 'HINSCAID_c',
            'HINSCARE': 'HINSCARE_c',
            'EDUC': 'EDUC_c',
            'EMPSTAT': 'EMPSTAT_c',
            'EMPSTATD': 'EMPSTATD_c',
            'LABFORCE': 'LABFORCE_c',
            'WRKLSTWK': 'WRKLSTWK_c',
            'ABSENT': 'ABSENT_c',
            'LOOKING': 'LOOKING_c',
            'AVAILBLE': 'AVAILBLE_c',
            'WRKRECAL': 'WRKRECAL_c',
            'WORKEDYR': 'WORKEDYR_c',
            'INCTOT': 'INCTOT_n',
            'INCWAGE': 'INCWAGE_n',
            'INCINVST': 'INCINVST_n',
            'POVERTY': 'POVERTY_n',
            # 'PERWT': 'PERWT_x',
            'ARRIVES': 'ARRIVES_x',
            # 'INCEARN': 'INCEARN_n',
            # 'INCWELFR': 'INCWELFR_n',
            'DEPARTS': 'DEPARTS_x',
            'sim_individual_id': 'sim_individual_id_i',
            }
'''
check_input - Checks the formatted ground truth data, the combination dictionary
and the number dictionary for proper formatting and values.

df - the formated ground truth dataframe
combo_dict - the combination dictionary
num_dict - the number dictionary

returns result - 1 if True, 0 if False
'''
def check_input(df, combo_dict, num_dict):
    result = 0
    int_result = 0
    pos_result = 1
    idx_result = 0
    cat_result = 0
    col_result = 0
    com_result = 1
    key_result = 1
    num_result = 1
    col_list = list(df.columns)
    col_len = len(col_list)

    # Check for all integers
    types = (df.dtypes <= np.int64) | (df.dtypes <= np.int32)
    all_ints = pd.Series(types.all())
    if all_ints.bool():
        int_result = 1
    else:
        logger.info("All values in dataframe must be integers")
        return 0

    for col in col_list:

        # Check for positive integers
        neg_numbers = (df[col] < 0).any()
        if neg_numbers:
            pos_result = 0
            logger.info(f"{col} contains negative numbers")

        # Check column names, _i, _c, _n, _x
        if col[-2:] == '_i':
            idx_result += 1
        if (col[-2:] == '_c') | (col[-2:] == '_n') | (col[-2:] == '_x'):
            cat_result += 1

    if idx_result != 1:
        logger.info("The dataframe must contain exactly one individual column ending in _i")
    else:
        idx_result = 1
    if (idx_result + cat_result) != col_len:
        logger.info("The dataframe columns are not labeled properly")
    else:
        col_result = 1

    for key in combo_dict:
        for i in range(len(combo_dict[key])):
            dkey = combo_dict[key][i]
            if dkey not in col_list:
                com_result = 0
                logger.info(f"The column value {dkey} in the combo_dict does not exist in the dataframe")

    for key in num_dict:
        if key not in col_list:
            key_result = 0
            logger.info(f"The column value {key} in the num_dict does not exist in the dataframe")
        if len(num_dict[key]) != 3:
            num_result = 0
            logger.info("There must be three parameters in the num_dict value list")
        top_value = num_dict[key][0]
        inc = num_dict[key][1]
        max_range = num_dict[key][2]
        if inc >= top_value:
            num_result = 0
            logger.info(f"the increment parameter {inc} must be less than the top value {top_value}")
        if max_range < top_value:
            num_result = 0
            logger.info(f"the max range parameter {max_range} must be greater than the top value {top_value}")

    if ((int_result == 1) & (pos_result == 1) &
        (col_result == 1) & (com_result == 1) &
        (key_result == 1) & (num_result == 1)):
        result = 1

    return result


'''
convert_codes - Used in convert_num to convert a number to an integer code.

n - a float or integer
d - a code dictionary

returns an integer
'''
def convert_codes(n, d):
    if n < 1:
        return 10
    else:
        for key in d:
            if key >= n:
                return int(d[key])
                break


'''
convert_cat - Used in preprocess to convert categorical features into codes.

df - the formated ground truth dataframe

returns df - a new dataframe with all _c columns converted to _char coded columns
'''
def convert_cat(df):
    col_list = list(df.columns)
    for col in col_list:
        if (col[-2:] == '_c'):
            max_value = df[col].max()
            fill_value = len(str(max_value))
            new_col = col + '_char'
            df[new_col] = df[col].astype(str).apply(lambda x: x.zfill(fill_value))
    return df


'''
convert_num - Used in preprocess to convert numerical features into codes.

df - the formated ground truth dataframe
num_dict - the number dictionary

returns df - a new dataframe with all _n columns converted to _char coded columns
returns num_code - a dictionary of number codes
returns num_decode - a dictionary for decoding numbers
'''
def convert_num(df, num_dict):
    col_list = list(df.columns)
    num_code = {}
    num_decode = {}
    for col in col_list:
        if (col[-2:] == '_n'):
            start_range = 0
            base_range = num_dict[col][0] + num_dict[col][1]
            end_range = num_dict[col][1]
            total_range = np.arange(start_range, (base_range), end_range)

            bins = np.r_[-np.inf, total_range, np.inf]
            population = [i + 10 for i in range(len(bins))]
            population.pop()

            code_dict = {bins[i+1]: population[i] for i in range(len(population))}
            key = col
            value = code_dict
            num_code[key] = value

            decode_dict = {population[i]: bins[i+1] for i in range(len(population))}
            key = col
            value = decode_dict
            num_decode[key] = value

            new_col = col + '_char'
            df[new_col] = df[col].apply(convert_codes, args=[code_dict])
            df[new_col] = df[new_col].astype(str)
    return df, num_code, num_decode


'''
combine_cols - Used in preprocess to combine columns based on the combo dictionary.

df - the formatted and converted ground truth dataframe
combo_dict - the combination dictionary

returns df - a formatted, converted and combined ground truth dataframe
returns col_decode - a dictionary for decoding the combined columns
'''
def combine_cols(df, combo_dict):
    col_decode = {}
    val_list = []
    for key in combo_dict:
        for i in range(len(combo_dict[key])):
            dkey = combo_dict[key][i]
            col = combo_dict[key][i] + '_char'
            if i == 0:
                if col[-6:] == 'c_char':
                    df[key] = '1' + df[col]
                    start = 1
                    end = len(df[col][0]) + start
                else:
                    df[key] = df[col]
                    start = 0
                    end = len(df[col][0]) + start
            else:
                df[key] = df[key] + df[col]
                start = end
                end += len(df[col][0])
            val_list.append(key)
            val_list.append(start)
            val_list.append(end)
            col_decode[dkey] = val_list
            val_list = []
        df[key] = df[key].astype(int)

    df = df.drop([col for col in df.columns if col.endswith('_char')], axis=1)
    df = df.drop([col for col in df.columns if col.endswith('_c')], axis=1)
    df = df.drop([col for col in df.columns if col.endswith('_n')], axis=1)

    return df, col_decode


'''
preprocess - Pre-processes the formatted ground truth by combining the columns
per the combination dictionary and the number dictionary.

df - the formatted ground ground_truth
combo_dict - the combination dictionary
num_dict - the number dictionary

returns df - a formatted, converted and combined ground truth dataframe
returns num_decode - a dictionary for decoding numbers
returns col_decode - a dictionary for decoding the combined columns
'''
def preprocess(ground_truth, combo_dict, num_dict):
    df = convert_cat(ground_truth)
    df, num_code, num_decode = convert_num(df, num_dict)
    df, col_decode = combine_cols(df, combo_dict)
    return df, num_decode, col_decode


'''
laplaceMechanism - A function for adding random noise using the Laplace Mechanism.

x - the input number
m - the sensitivity value
e - the epsilon value
'''
def laplaceMechanism(x, m, e):
    if x != 0:
        x += np.random.laplace(0, m/e, 1)[0]
    return x


'''
weight - Used in create_private_histo to calculate the privatized histogram weights.

df - the preprocessed dataframe
col - the column to calculate weights
bins - a list of bins to store the weights
sample - 1 if using sampling, 0 if no sampling
sample_size - the sample size
sensitivity - the sensitivity value
epsilon - the epsilon value

returns wt - a list of weights
'''
def weight(df, col, bins, sensitivity, epsilon):
    new_df = df[col].groupby(pd.cut(df[col], bins)).count()

    raw_list = new_df.values.tolist()
    noisy_list = [laplaceMechanism(x, sensitivity, epsilon) for x in raw_list]
    zeroed_list = [x if x > 0 else 0 for x in noisy_list]

    wt = [x / sum(zeroed_list) for x in zeroed_list]

    return wt


'''
histo_test - Counts the number of bins created for each combined column in the
preprocessed dataframe.  A useful tool for determining the optimal feature
combinations.

df - the preprocessed dataframe
combo_dict - the combination dictionary

prints a count of bins for each combined column
'''
def histo_test(df, combo_dict):
    for key in combo_dict:
        bins = df[key].unique().tolist()
        logger.info(f"number of bins in {key} {len(bins)}")

# Creates the weights for the bins adding noise to the bin counts
# using the Laplace mechanism (for the individual histograms)
#    df - the input dataframe
#    c - the column
#    b - the bins (population)
#    s - sensitivity
#    e - epsilon
#    return the weights for the population
def weight_x(df, c, b, s, e):
    new_df = df[c].groupby(pd.cut(df[c], b)).count()

    for i in range(len(new_df)):
        new_df.iloc[i] = laplaceMechanism(new_df.iloc[i], s, e)

    final_df = new_df / new_df.sum()
    return new_df.values


#     print(new_df)
#     w = []
#     for i in range(len(final_df.index)):
#         w.append(final_df[i])
#     return w

'''
create_private_histo - Created privatized histograms of the specified combined
column using a specified sample size, sensitivity and epsilon.

df - the preprocessed dataframe
col - the column
sample - 1 if using sampling, 0 if no sampling
sample_size - the sample size
sensitivity - the sensitivity value
epsilon - the epsilon value
'''
def create_private_histo(df, col, bins, sensitivity, epsilon):
    renamed_col = col_dict[col]
    wt = weight_x(df, renamed_col, bins, sensitivity, epsilon)
    return wt

def create_private_histo_combined(df, col, sensitivity, epsilon):
        bins = df[col].unique().tolist()
        bins.append(-np.inf)
        bins.sort()

        pop = df[col].unique().tolist()
        pop.sort()

        wt = weight(df, col, bins, sensitivity, epsilon)

        return pop, wt


'''
col_decoder - Decodes the specified column into its privatized value.

num_dict - the number dictionary
num_decode - a dictionary used to decode the numbers
col_decode - a dictionary used to decode the columns
combined_value - the combined value to decode
col - the column to decode

returns result - a decoded privatized value
'''
def col_decoder(num_dict, num_decode, col_decode, combined_val, col):
    start = int(col_decode[col][1])
    end = int(col_decode[col][2])
    number = int(str(combined_val)[start:end])
    if (col[-2:] == '_n'):
        top_value = num_dict[col][0]
        inc = num_dict[col][1]
        max_range = num_dict[col][2]
        top_range = (top_value/inc) + 11

        if number == top_range:
            v = top_value
        else:
            v = num_decode[col][number]

        if v == 0:
            value = 0
        elif v == top_value:
            value = random.randrange(top_value, max_range)
        else:
            value = random.randrange(v - inc, v)
        result = value
    else:
        result = number
    return result

def rename_combo_dict_keys(combo_dict):
    renamed = {}
    for key in combo_dict.keys():
        temp = []
        for value in combo_dict[key]:
            temp.append(col_dict[value])
        renamed[key] = temp
    return renamed

def combine_columns(ground_truth, combo_dict, num_dict):
    ground_truth = ground_truth.drop(['PERWT', 'INCEARN', 'INCWELFR'], axis=1)
    combo_dict = rename_combo_dict_keys(combo_dict)
    num_dict = {col_dict[key]: num_dict[key] for key in num_dict.keys()}
    # Encode strings as integers
    ground_truth['PUMA'] = ground_truth['PUMA'].str.replace('-', '')
    ground_truth['PUMA'] = ground_truth['PUMA'].astype(int)
    ground_truth['HHWT'] = ground_truth['HHWT'].astype(int)
    # ground_truth['PERWT'] = ground_truth['PERWT'].astype(int)
    ground_truth[ground_truth < 0] = 0
    ground_truth = ground_truth.rename(columns=col_dict)
    logger.info("begin pre-processing")
    valid = check_input(ground_truth, combo_dict, num_dict)
    if valid != 1:
        print('invalid')
        return
    df, num_decodes, col_decodes = preprocess(ground_truth, combo_dict, num_dict)
    histo_test(df, combo_dict)
    logger.info("end pre-processing")
    return df, num_decodes, col_decodes
