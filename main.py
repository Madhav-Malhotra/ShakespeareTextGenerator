import tensorflow as tf

import numpy as np
import os
import time
import functools

print("Imported libraries");

#Download data
filePath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt');
# Read, then decode for py2 compat.
text = open(filePath, 'rb').read().decode(encoding = 'utf-8');

#Text length is number of characters
print ('Length of text: {} characters'.format(len(text)));
#The unique characters in the text
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)));

#Maps characters to ints (65 unique ints)
charIndex = {uniqueChar: num for num, uniqueChar in enumerate(vocab)};
#Maps ints to characters (65 unique characters)
indexChar = np.array(vocab);
#Shakespearean text represented as ints.
intText = np.array([charIndex[char] for char in text]);

#Conversion of first 13 characters to ints
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), intText[:13]));

#Each input is 100 characters maximum
sequenceLength = 100;
#The number of batches used in each epoch.
examplesPerEpoch = len(text) // sequenceLength;

# Create training examples/targets
charDataset = tf.data.Dataset.from_tensor_slices(intText);
#Joins characters into 100-char sequences
sequences = charDataset.batch(sequenceLength+1, drop_remainder = True);

def splitInputTarget(chunk):
  '''
  Duplicates sequence of chars and shifts right by 1
  Input: Tensor [-1, 100]
  Output: Tensors [-1, 100]
  '''
  inputText = chunk[:-1];
  targetText = chunk[1:];
  return inputText, targetText;

dataset = sequences.map(splitInputTarget);

print("Prepared Data");

#Batch size of training examples 
batchSize = 64;
#Number of batches run through per epoch
stepsPerEpoch = examplesPerEpoch//batchSize;
# Buffer size to shuffle the dataset
#Note: Shuffles 'buffer' of elements at time
#in case of extremely large number of elements
bufferSize = 10000
#Shuffles dataset buffer
dataset = dataset.shuffle(bufferSize).batch(batchSize, drop_remainder = True);

print("Shuffled Data");

#Input + Output layer's dimensions
vocabSize = len(vocab);
#Embedding layer's dimension 
embeddingDim = 256;
#RNN Layer's Dimensions
rnnUnits = 1024;

#Creating RNN without GPU processing
rnn = functools.partial(
  tf.keras.layers.GRU, recurrent_activation='sigmoid');

print("Functools RNN");

def buildModel(vocabSize, embeddingDim, rnnUnits, batchSize):
  '''
  Creates a RNN network
  Input: int(vocabSize) = Input and output data dimensions
  Input: int(embeddingDim) = Input layer's output mapping dimensions
  Input: int(rnnUnits) = 
  '''
  model = tf.keras.Sequential([
    #Creates input layer.
    tf.keras.layers.Embedding(vocabSize, embeddingDim, 
      batch_input_shape = [batchSize, None]),
    #Creates RNN layer
    rnn(rnnUnits,
      return_sequences = True, 
      recurrent_initializer = 'glorot_uniform',
      stateful = True),
     #Creates ouput layer 
    tf.keras.layers.Dense(vocabSize)
  ]);
  return model;

#Creates network
net = buildModel(
  vocabSize = len(vocab), 
  embeddingDim = embeddingDim, 
  rnnUnits = rnnUnits, 
  batchSize = batchSize);

print("Built Network");
net.summary();

def loss(labels, logits):
  '''
  Establishes loss Function
  '''
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True);

#Setting up network
net.compile(
    optimizer = tf.compat.v1.train.AdamOptimizer(),
    loss = loss);

# Directory where the checkpoints will be saved
checkpointDir = './training_checkpoints';
# Name of the checkpoint files
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}");
#Saves training progress during training.
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpointPrefix,
    save_weights_only = True);

#Number of epochs trained
EPOCHS = 30;
#Training network
history = net.fit(dataset.repeat(), epochs = EPOCHS, steps_per_epoch = stepsPerEpoch, callbacks = [checkpointCallback]);

#Restoring neuron weights from last checkpoint
tf.train.latest_checkpoint(checkpointDir);

#Restoring model given previous training
model = buildModel(vocabSize, embeddingDim, rnnUnits, batch_size = 1);
model.load_weights(tf.train.latest_checkpoint(checkpointDir));
model.build(tf.TensorShape([1, None]));


def generateText(model, startString):
  '''
  Evaluation step for model 
  (generates Shakespearean text)
  Input: RNN used
  Input: String to start off with (Ex. 'C' or 'ROMEO: ')
  Output: String - text generated
  '''

  # Number of characters to generate
  numGenerate = 1000

  # Converting start string to numbers (vectorizing) 
  inputEval = [charIndex[char] for char in startString];
  inputEval = tf.expand_dims(inputEval, 0);

  #Initialise output string
  textGenerated = [];

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  temperature = 1.0;

  # Here batch size == 1
  model.reset_states()
  for char in range(numGenerate):
      predictions = model(inputEval);
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0);

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature;
      predictedId = tf.multinomial(predictions, num_samples = 1)[-1,0].numpy();
      
      #Word generated and current hidden state 
      #sent as input for model's next prediction
      inputEval = tf.expand_dims([predictedId], 0)
      
      textGenerated.append(indexChar[predictedId]);
  #Outputs generated text
  return (startString + ''.join(textGenerated));

#--------------------------------------------------
#Customised training to make results more predictable

#Setting up model
model = buildModel(
  vocab_size = len(vocab), 
  embedding_dim = embeddingDim, 
  rnn_units = rnnUnits, 
  batch_size = batchSize);

#Optimiser function (to find weight, bias adjustments)
optimiser = tf.compat.v1.train.AdamOptimizer();

# Training step
EPOCHS = 1;

for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing hidden state before every epoch
    # initally hidden is None
    hidden = model.reset_states();
    
    #Goes through dataset, batch by batch
    for (batch_n, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # feeding the hidden state back into the model
            #Prediction and loss calculations
            predictions = model(inp);
            loss = tf.losses.sparse_softmax_cross_entropy(target, predictions);

        #Calculating gradients of loss    
        grads = tape.gradient(loss, model.trainable_variables);
        optimiser.apply_gradients(zip(grads, model.trainable_variables));

        #Outputting performance every 100 batches
        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {:.4f}'
            print(template.format(epoch+1, batch_n, loss));

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      model.save_weights(checkpointPrefix.format(epoch = epoch));

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss));
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start));

#Saving weights for recovering training progress.
model.save_weights(checkpointPrefix.format(epoch = epoch));