# From raw texts to embeddings
library(keras)

# PART 1
# Download the raw IMDB reviews
# dataset downloaded from http://mng.bz/0tIo
# Process the data
imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")

labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names=TRUE)) {
      texts <- c(texts, readChar(fname, file.info(fname)$size))
      labels <- c(labels, label) 
       }
}

# PART 2 
# Tokenizing the data and Preparing training and validation
# limit the training data to the first 200 reviews (rather than 25000)
# we will use pre-embeddings

maxlen <- 100 #cuts reviews after 100 words
training_samples <- 200 # trains only 200 samples
validation_samples <- 10000 
max_words <- 10000 #consider only 10000 words from the dataset

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)

word_index <- tokenizer$word_index

# pad i.e., cut off at one-hyndred (convert from list to
# array)
data <- pad_sequences(sequences, maxlen=maxlen)

# Convert from atomic vector (1d) to array (nd)
labels <- as.array(labels)

# Reorder indices
indices <- sample(1:nrow(data))


training_indices <- indices[1:training_samples]
validation_indices <- indices[training_samples + 1:training_samples + 
                                validation_samples]

x_train <- data[training_indices,]
y_train <- labels[training_indices]

x_val <- data[validation_indices,]
y_val <- data[validation_indices]

# PART 3 
# Preprocessing the Embeddings
# download the GloVe precomputed word embeddings from 2014 English Wikipedia
# https:/nlp.stanford.edu/projects/glove

glove_dir <- "~/Downloads/glove.6B"
# read the 100-dimensional vectors into memory
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

# Create index that maps words (as strings to their vector representation as numeric vectors)
# new environment
embeddings_index <- new.env(hash=TRUE, parent=emptyenv())
for (i in 1:length(lines)) {
  line <- lines[i]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

# Map onto a matrix of size (max_words, embedding_dim), where i is the index of the word
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))

for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
        embedding_matrix[index + 1,] <- embedding_vector
  }
}
