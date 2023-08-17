



# labels, i just did this on the fly
labels = {
    'stable': (-5, 5),
    'loss': (-25, -5),
    'high loss': (-100, -25),
    'gain': (5, 25),
    'high gain': (25, 100)
}

# Function to assign label based on value
def assign_label(value):
    for label, (lower, upper) in labels.items():
        if lower <= value <= upper:
            return label
    return 'out of range'

prognostic_value = round(dict_fit_VAR_output["avg_last_10_days_diff"])

# Calculate the label for the prognostic_value
label = assign_label(prognostic_value)

# Function to assign label based on value
def assign_label(value):
    for label, (lower, upper) in labels.items():
        if lower <= value <= upper:
            return label
    return 'out of range'

# Calculate the label for the prognostic_value
label = assign_label(prognostic_value)

# TODO fix train and test data to be able to plot them
# Visualize results for the 'close' column
plt.figure(figsize=(10, 6))
plt.plot(train_data['close'], label='Training Data')
plt.plot(test_data.index, forecast[:, df_selected.columns.get_loc('close')], color='green', label='Forecast')
plt.plot(test_data.index, test_data['close'], color='red', label='Actual Data')
plt.xlabel('Date')
plt.ylabel('Close Value')
plt.title(f'VAR Forecast for BTC-USD, MAPE: {round(mape,2)}%\nAverage last 3 days diff: {round(average_last3_days_diff ,1)}%\n Verdict: {label}  ') 
plt.legend()
plt.show()