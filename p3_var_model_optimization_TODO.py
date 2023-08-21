from dateutil.relativedelta import relativedelta

# Initialize variables to keep track of the best configuration
best_mape = float('inf')
best_avg_last3days = float('inf')
best_cols = []
best_order = None

# Iterate over potential columns and order values
"""for cols in range(2, len(potential_columns) + 1):
    for order in range(1, 40, 5):  # goes through the range in steps of 5, adjust as needed
        
        # Iterate over different train data lengths 
        for start_date in pd.date_range(start='2019-04-15', end=train_end_date - relativedelta(months=6) , freq='1M'):  # intervall set to 1 month, adjust as needed, at least 6 months
"""







"""
assuming we did the for loop above

# Print the best configuration
print("Best Features to use: ", best_cols)
print("Best Order Value:", best_order)
#print("Best MAPE:", best_mape)

#print("Best avg last3days:", best_avg_last3days:)
print("Best Train data range:", best_start_date, end_date)
print("Best Average last3 days diff %:", best_avg_last3days)"""