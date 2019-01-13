import argparse
import zipfile
import re
import collections
import numpy as np
from six.moves import xrange
import random
import torch
from torch.autograd import Variable
from models import SkipGramModel
from models import CBOWModel


cmd_parser = argparse.ArgumentParser(description=None)
# Data arguments
cmd_parser.add_argument('-d', '--data', default='data/text8.zip',
                        help='Data file for word2vec training.')
cmd_parser.add_argument('-o', '--out', default='word2vec.json',
                        help='Output filename.')
cmd_parser.add_argument('-p', '--plot', default='tsne.png',
                        help='Plotting output filename.')
cmd_parser.add_argument('-pn', '--plot_num', default=100, type=int,
                        help='Plotting data number.')
cmd_parser.add_argument('-s', '--size', default=50000, type=int,
                        help='Vocabulary size.')
# Model training arguments
cmd_parser.add_argument('-bs', '--batch_size', default=128, type=int,
                        help='Training batch size.')
cmd_parser.add_argument('-ns', '--num_skips', default=2, type=int,
                        help='How many times to reuse an input to generate a label.')
cmd_parser.add_argument('-sw', '--skip_window', default=1, type=int,
                        help='How many words to consider left and right.')
cmd_parser.add_argument('-ed', '--embedding_dim', default=128, type=int,
                        help='Dimension of the embedding vector.')
cmd_parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate')
cmd_parser.add_argument('-i', '--num_steps', default=10000, type=int,
                        help='Number of steps to run.')
cmd_parser.add_argument('-n', '--num_sampled', default=64, type=int,
                        help='Number of negative examples to sample.')


# Read the data into a list of words.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as f:
            text = f.read(f.namelist()[0]).decode('ascii')
    else:
        with open(filename, "r") as f:
            text = f.read()
    return [word.lower() for word in re.compile('\w+').findall(text)]

def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
        Returns:
            data        list of codes (integers from 0 to vocabulary_size-1).
                        This is the original text but words are replaced by their codes
            count       map of words(strings) to count of occurrences
            dictionary  map of words(strings) to their codes(integers)
            reverse_dictionary  maps codes(integers) to words(strings)
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    """Generates a batch of training data
        returns:
            batch:      a list of data index for this batch.
            labels:     a list of contexts indexes for this batch.
            data_index: current data index for next batch.
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return torch.LongTensor(batch), torch.LongTensor(labels), data_index


def train(data, vocabulary_size, embedding_dim, batch_size, num_skips, skip_window, num_steps, learning_rate):
    model = CBOWModel(vocabulary_size, embedding_dim)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate)
    data_index = 0
    loss_val = 0
    loss_function = torch.nn.NLLLoss()
    for i in xrange(num_steps):
        # prepare feed data and forward pass
        batch, labels, data_index = generate_batch(data, data_index,
            batch_size, num_skips, skip_window)
        y_pred = model(labels)
        loss = loss_function(y_pred, batch)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss value at certain step
        loss_val += loss.item()
        if i > 0 and i % (num_steps/100) == 0:
            print('Average loss at step', i, ':', loss_val/(num_steps/100))
            loss_val = 0

    return model.get_embeddings()


def tsne_plot(embeddings, labels, num, reverse_dictionary, filename):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to plot embeddings.')
        print(ex)
        return
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(final_embeddings[:num, :])
    low_dim_labels = [reverse_dictionary[i] for i in xrange(num)]
    assert low_dim_embs.shape[0] >= len(low_dim_labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(low_dim_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    print("saving plot to:", filename)
    plt.savefig(filename)

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    # Data preprocessing
    vocabulary = read_data(args.data)
    print('Data size', len(vocabulary))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                args.size)
    vocabulary_size = min(args.size, len(count))
    print('Vocabulary size', vocabulary_size)
    # Model training
    final_embeddings = train(data=data,
                             vocabulary_size=vocabulary_size,
                             embedding_dim=args.embedding_dim,
                             batch_size=args.batch_size,
                             num_skips=args.num_skips,
                             skip_window=args.skip_window,
                             num_steps=args.num_steps,
                             learning_rate=args.learning_rate)
    norm = torch.sqrt(torch.cumsum(torch.mul(final_embeddings, final_embeddings), 1))
    nomalized_embeddings = (final_embeddings/norm).numpy()
    # Save result and plotting
    tsne_plot(embeddings=nomalized_embeddings,
              labels=labels,
              num=min(vocabulary_size, args.plot_num),
              reverse_dictionary=reverse_dictionary,
              filename=args.plot)