source("scripts/kappalab_func.R")

server <- function() {
  while(TRUE){
    # writeLines("Listening...")
    con <- socketConnection(host="localhost", port = 6011, blocking=TRUE,
                            server=TRUE, open="r+", timeout = 3600)
    input <- fromJSON(readLines(con, 1))
    tryCatch({
      output <- main(input)
      writeLines(toJSON(output), con)
      close(con)
      #print("success")
      #print(output)
    },
    error = function(error_msg) {
      json_res <- list(errorMessages = error_msg$message)
      # print(toJSON(json_res))
      writeLines(toJSON(json_res), con)
      close(con)
      #print("failure")
    },
    finally = function(f) {
      close(con)
      print("close connection")
    })
  }
}
server()