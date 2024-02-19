import pandas as pd
from itertools import combinations


MIN_SUPPORT = 0.50
CONFIDENCE = 50

def get_transactions(df):
    columns = df.columns
    
    transactions = []
    
    for _, row in df.iterrows():
        transaction = set()
        
        for col in columns:
            if row[col] == "Yes":
                transaction.add(col)

        transactions.append(transaction)
    
    return transactions

def get_support_count(transactions, item_sets):
    count = {}
    
    for item_set in item_sets:                
        item_set = frozenset(item_set)
        for transaction in transactions:
            if item_set.issubset(transaction):
                if item_set in count:
                    count[item_set] += 1
                else:
                    count[item_set] = 1
             
    support_count_df = pd.DataFrame()
    support_count_df['item_sets'] = count.keys()
    support_count_df['support_count'] = count.values()
    return support_count_df

def remove_item_sets(support_count_df, min_support_count):
    support_count_df = support_count_df[support_count_df['support_count'] >= min_support_count]
    return support_count_df

def generate(item_sets, k):
    new_item_sets = set()
    counter = 1
    for item_set1 in item_sets:
        for item_set2 in item_sets[counter:]:
            merged_item_set = item_set1.union(item_set2)
            if len(merged_item_set) == k:
                new_item_sets.add(merged_item_set)
        counter += 1
    
    return list(new_item_sets)

def apriori(transactions, min_support_count):
    item_sets = [{col} for col in df.columns]
    k = 1
    
    support_count_dict = {}
    support_count_df = get_support_count(transactions, item_sets)
    
    result = support_count_df
    
    while len(support_count_df) != 0:
        support_count_df = remove_item_sets(support_count_df, min_support_count)
        support_count_dict[k] = support_count_df
        
        if len(support_count_df) > 1 or len(support_count_df) == 1 and support_count_df.iloc[0].support_count >= min_support_count:
            result = support_count_df

        k += 1
        item_sets = generate(support_count_df['item_sets'], k)
        support_count_df = get_support_count(transactions, item_sets)
    
    return result, support_count_dict

def generate_rules(frequent_item_set):
    rules = []
    
    for item_set in frequent_item_set.item_sets:
        for count in range(1, len(item_set)):
            combinations_list = list(combinations(item_set, count))
            for combination in combinations_list:
                consequent = [item for item in item_set if item not in combination]
                rules.append(list(combination))
                rules.append(consequent)
                
    return rules

def get_confidence(antecedent, consequent, support_count_dict):
    numerator, denominator = 0, 0
    
    support_count_df = support_count_dict[len(antecedent)]

    for i in range(len(support_count_df)):
        if set(support_count_df.iloc[i, 0]).difference(antecedent) == set():
            numerator = support_count_df.iloc[i, 1]
            break
    
    support_count_df = support_count_dict[len(consequent)]
    
    for i in range(len(support_count_df)):
        if set(support_count_df.iloc[i, 0]).difference(consequent) == set():
            denominator = support_count_df.iloc[i, 1]
            break
    
    return numerator/denominator*100

if __name__ == "__main__":
    df = pd.read_csv("Covid_Dataset.csv")
    df = df[df["COVID-19"]=="Yes"]
    df = df[df.columns[:-1]]

    transactions = get_transactions(df)
    min_support_count = MIN_SUPPORT * len(df)
    print("MINIMUM SUPPORT COUNT:", min_support_count, "\n")
    
    frequent_item_set, support_count_dict = apriori(transactions, min_support_count)
    print(frequent_item_set.to_string(), "\n")
    
    rules = generate_rules(frequent_item_set)
    
    for i in range(0, len(rules), 2):
        confidence = get_confidence(set(rules[i]).union(set(rules[i+1])), set(rules[i]), support_count_dict)
        if confidence > CONFIDENCE:
            print(rules[i], "->", rules[i+1], ":", confidence)
    