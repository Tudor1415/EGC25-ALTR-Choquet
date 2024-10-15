source("scripts/kappalab_func.R")

argv <- commandArgs(trailingOnly = TRUE)
input_file <- argv[1]
input <- fromJSON(input_file)
output <- main(input)
write_json(output, argv[2])