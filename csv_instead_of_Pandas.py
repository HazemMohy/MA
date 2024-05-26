# Prepare data for CSV using csv (instead of pandas)
epochs = list(range(1, max_epochs + 1)) #Generate a list of epoch numbers.
training_losses = epoch_loss_values #Extract training loss values from epoch_loss_values.
validation_metrics = [None] * max_epochs #Initialize a list for validation metrics with None values for each epoch.

#For each validation metric in metric_values, place it in the corresponding position in the validation_metrics list based on the validation interval.
for i, metric in enumerate(metric_values): 
    validation_epoch = (i + 1) * val_interval
    validation_metrics[validation_epoch - 1] = metric

# Create a self-explanatory CSV file name
csv_file_name = f"tracking_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}_{test_metric_value:.4f}.csv"
csv_file_path = os.path.join(tracking_dir, csv_file_name)

#Open a CSV file for writing.
with open(csv_file_path, mode='w', newline='') as file: 
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Average Training Loss", "Average Validation Dice Metric"]) #Write the header row.

    #Write the epoch, average training loss, and average validation Dice metric for each epoch
    for epoch, train_loss, val_metric in zip(epochs, training_losses, validation_metrics):
        writer.writerow([epoch, train_loss, val_metric])

    #Add rows for the best evaluation metric and test Dice metric at the end of the file.
    writer.writerow([])
    writer.writerow(["Best Evaluation Metric", best_metric])
    writer.writerow(["Test Dice Metric", test_metric_value])

print(f"Metrics CSV saved to {csv_file_path}")
