"""GPT2 code to extract predictions and embeddings from podcast file

We feed in a sequence of tokenized text. Something like:

    ... The quick brown fox jumped over the lazy dog.

In order to get the probability the word dog was predicted from its context, we
look at the output of the _predictions_ at the the time point of lazy. However,
to get the representation for the word dog, we take the hidden state of the
word dog.
"""

# Initial code by Z. Zaida.
# M. Kumar added code to compute surprise and entropy

import argparse
import json
import os
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from lcs import lcs

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--datum-file', type=str, required=True)
parser.add_argument('--sentence-file', type=str, required=True)
parser.add_argument('--story-name', type=str, required=True)
parser.add_argument('--context-length', type=int)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save-predictions', action='store_true')
parser.add_argument('--save-hidden-states', action='store_true')
parser.add_argument('--use-previous-state', action='store_true')
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                          add_prefix_space=True,
                                          cache_dir="/scratch/gpfs/mk35/.cache/transformers/")
if args.context_length <= 0:
    args.context_length = tokenizer.model_max_length
assert args.context_length <= tokenizer.model_max_length, \
    'given length is greater than max length'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(args)

# Load pre-trained model
save_states = args.save_hidden_states
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             output_hidden_states=save_states)
model = model.to(device)
model.eval()  # Set the model in evaluation mode to deactivate dropout modules


base_name = f'{args.story_name}{args.model_name}-c_{args.context_length}{args.suffix}'
os.makedirs(os.path.join('results', base_name), exist_ok=True)

# Read all words and tokenize them
# NOTE - I had to strip the newline in order to reproduce previous embeddings
#        maybe the old tokenizer did it for us?
tokens = []
if args.story_name in ('pieman', 'piemandouble','tunnel'):
    sentence_file= '%s_transcript.txt' % args.story_name
with open(args.sentence_file, 'r') as fp:
    for line in fp:
        tokens.extend(tokenizer.tokenize(line.rstrip()))

# Convert to indices
token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens),
                         device=device)
assert len(tokens) == len(token_ids)

data = []
ranks = []
predictions = []
predicted_words = []
predicted_probs = []
target_probs = []
df_sur_entr=pd.DataFrame(columns={'Target Word','Target Word Probability','Surprise', 'Entropy','Top 1 Probability', 'Top 1 Word','Top 5 Probability', 'logits' })
iter_fun = range if args.verbose else tqdm.trange
for i in iter_fun(1, len(tokens)-1):

    # Skip punctuation
    if tokens[i] in string.punctuation:
        continue

    # TODO - use the past variable to make it quicker
    # token_ids is the list of tokens.
    start, end = max(0, i-args.context_length+1), i + 1
    context = token_ids[start:end].unsqueeze(0)

    if args.verbose:
        print('Encoding "%s" from sequence:\t' %
              (tokenizer.decode(token_ids[i].tolist())),
              tokenizer.decode(token_ids[end-10:end].tolist()), end='|')

    # outputs is a tuple of:
    # prediction_scores: (batch_size, sequence_length, config.vocab_size)
    # past: unused
    # hidden_states: [(batch_size, sequence_length, hidden_size)]
    with torch.no_grad():
        outputs = model(context)

    # Get the probability for the _correct_ word
    prediction_scores = outputs[0]
    probabilities = F.softmax(prediction_scores[0, -2], dim=0)  # Note: -2
    word_probability = probabilities[token_ids[i]].item()
   
   #Compute Surprise and Entropy

    logp = np.log2(probabilities.cpu()).numpy()
    prob=probabilities.cpu().numpy()
    df_sur_entr.loc[i,'Target Word Probability']=word_probability
    df_sur_entr.loc[i,'Surprise']= -np.log2(word_probability)
    df_sur_entr.loc[i,'Entropy']= -np.sum(prob*logp)
    
    # Use gpt2 vocab for top1
    predicted_word = tokenizer.decode(probabilities.argmax().item())

    true_id = context[0][-1].item()
    rank = torch.nonzero(true_id == probabilities.argsort(descending=True))
    ranks.append(rank.item())

    if args.save_predictions:
        predictions.append(prediction_scores[0, -2].unsqueeze(0).cpu())
    
        #Add the top1 predictions to the data frame
    df_sur_entr.loc[i,'Top 1 Probability']=max(probabilities.cpu()).numpy()
    df_sur_entr.loc[i,'Target Word']=tokenizer.decode(token_ids[i].tolist())
    df_sur_entr.loc[i,'Top 1 Word']=predicted_word 
    df_sur_entr.loc[i,'Top 5 Probability']=np.sort(probabilities.cpu().numpy())[-5:].tolist()
    df_sur_entr.loc[i,'logits']=prediction_scores[0, -2].cpu().tolist()

    if args.verbose:
        probsort, idxsort = probabilities.sort(dim=-1, descending=True)
        print(tokenizer.decode(idxsort[:10].tolist()),
              word_probability,
              predicted_word)

    # Get the hidden representation of the last word
    last_hidden_states = None
    if args.save_hidden_states:
        hidden_states = outputs[2]
        time_index = -2 if args.use_previous_state else -1
        # Just get the last hidden state, maybe change later
        last_hidden_states = [hidden_states[-1][0, time_index, :].cpu()]
        # last_hidden_states = [hidden_state[0,time_index,:].cpu() for
        #                      hidden_state in hidden_states]

    token = tokenizer.decode(token_ids[i].cpu().item()).strip().lower()
    data.append([token, word_probability, last_hidden_states, predicted_word])

# Save tokens and hidden states before aligning
with open(f'results/{base_name}/{base_name}.pkl', 'wb') as f:
    pickle.dump(data, f)
# If you want to debug...
# breakpoint()
# import pandas as pd
# df = pd.DataFrame.from_records([(e[0], e[2][0].tolist()) for e in data])

# Calculate manual accuracy
col1 = [d[0] for d in data]
col2 = [d[-1] for d in data]
acc1 = sum([x.strip() == y.strip() for x, y in zip(col1, col2)]) / len(col1)

# Align the two lists
df = pd.read_csv(args.datum_file, sep=',')
rows = list(row for _, row in df.iterrows())
words = list(map(str.lower, df.word.tolist()))
model_tokens = [d[0] for d in data]
mask1, mask2 = lcs(words, model_tokens)

aligned_data = []
for i, j in zip(mask1, mask2):
    data[j].append(rows[i])
    aligned_data.append(data[j])

print(f'Went from {len(data)} {len(words)} words to {len(aligned_data)} words')

assert len(aligned_data) > 0

# Calculate correlation with behavior
model_probs = np.array([d[1] for d in aligned_data])
mturk_probs = np.array([d[-1].cloze for d in aligned_data])
notnan = np.logical_not(np.isnan(mturk_probs))
model_probs = model_probs[notnan]
mturk_probs = mturk_probs[notnan]

# Save results
slope, intercept, r, p, std_err = stats.linregress(model_probs, mturk_probs)
results = {'r': r, 'p': p, 'std_err': std_err}

# Calculate manual accuracy
col1 = [d[0] for d in aligned_data]
col2 = [d[-2] for d in aligned_data]
acc = sum([x.strip() == y.strip() for x, y in zip(col1, col2)]) / len(col1)
results['accuracy'] = acc
results['accuracy1'] = acc1

# Calcualte top-k
ranks = np.array(ranks)
results['top1'] = (ranks == 0).sum() / len(ranks)
results['top5'] = (ranks < 5).sum() / len(ranks)
results['top10'] = (ranks < 10).sum() / len(ranks)

results['args'] = vars(args)
with open('results/' + base_name + '/results.json', 'w') as fp:
    json.dump(results, fp, indent=4)

plt.figure()
plt.scatter(mturk_probs, model_probs, marker='.', label='r=%.2f' % r, alpha=.3)
plt.plot(mturk_probs, mturk_probs * slope + intercept, '-k')
plt.ylabel('Model Probability')
plt.xlabel('Behavioral Cloze Probability')
plt.title(base_name)
plt.legend()
plt.savefig('results/' + base_name + '/plot.png', bbox_inches='tight')

# Write out predictions and probabilities
with open('results/' + args.story_name + '/predictions.csv', 'w') as fp:
    fp.write('datum word,token,token prob,predicted word\n')
    for token, word_prob, _, top_word, row in tqdm.tqdm(aligned_data):
        fp.write(f'{row.word},{token},{word_prob},{top_word}\n')

# Write out datum
with open('results/' + args.story_name + '/datum.csv', 'w') as fp:
    fp.write('token,onset,offset,prob,speaker,cloze,predicted_word\n')
    for token, word_prob, _, top_word, row in tqdm.tqdm(aligned_data):
        fp.write(f'{token},{row.onset},{row.offset},{word_prob},{row.speaker},{row.cloze},{top_word}\n')

# Write out each layer into a datum
if args.save_hidden_states:
    n_layers = len(aligned_data[0][2])
    output_files = [open('results/' + args.story_name + f'/{base_name}-layer_{i}.csv', 'w') for i in range(n_layers)]

    for struct in tqdm.tqdm(aligned_data, desc='outputing embedding files'):
        token = struct[0]
        word_prob = struct[1]
        pred_word = struct[3]
        last_hidden_states = struct[2]  # [(hidden_size)]
        row = struct[-1]  # pandas row
        for embedding, fp in zip(last_hidden_states, output_files):
            fp.write(f'{token},{row.onset},{row.offset},{word_prob},{row.speaker},')
            np.savetxt(fp, embedding.numpy(), newline=',', fmt='%.6f')
            fp.write('\n')

    for fp in output_files:
        fp.close()
#Save surprise and entropy values
output_file_surprise = f'results/{args.story_name}/{base_name}_surp_entr.csv'
df_sur_entr.iloc[mask2].to_csv(output_file_surprise)

# write probabilities into a .pt file
if args.save_predictions:
    predictions = torch.cat(predictions, dim=0).numpy()
    predictions = predictions[mask2, :]
    np.save('results/' + base_name + '/predictions.npy', predictions)

    model_tokens = [d[0] for d in aligned_data]
    np.save('results/' + base_name + '/words.npy', model_tokens)

    # save mask for human word alignment
    with open('results/' + base_name + '/human_mask.txt', 'w') as fp:
        for idx in mask1:
            fp.write('%s\n' % idx)
