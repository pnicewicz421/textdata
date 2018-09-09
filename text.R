# text data
library(keras)

samples <- c("The cat sat on the mat.", "The dog ate my homework")

tokenizer <- text_tokenizer(num_words=1000) %>%
            fit_text_tokenizer(samples)

sequences <- texts_to_sequences(tokenizer, samples)

one_hot_results <- texts_to_matrix(tokenizer, samples, mode="binary")