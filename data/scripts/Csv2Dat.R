# Function to convert CSV to DAT
convert_csv_to_dat <- function(input_file, output_file) {
  # Load the CSV file into R
  data <- read.csv(input_file, sep = ";")

  # Write the data to a .dat file
  write.table(data, file = output_file, sep = " ", col.names = FALSE,
              row.names = FALSE, quote = FALSE)
}

# Retrieve command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments is provided
if (length(args) != 2) {
  stop("Usage: Rscript script.R <input_csv_file> <output_dat_file>")
}

# Extract arguments
input_csv_file <- args[1]
output_dat_file <- args[2]

# Call the function to convert CSV to DAT
convert_csv_to_dat(input_csv_file, output_dat_file)
