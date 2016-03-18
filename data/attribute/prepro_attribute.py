"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import sys
import codecs
import locale
sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout) 
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def prepro_captions(imgs):
  
  # preprocess all the captions
  print 'example processed tokens:'
  for i,img in enumerate(imgs):
    img['processed_tokens'] = []
    for j,s in enumerate(img['captions']):
      txt = str(s).lower().translate(None, string.punctuation).strip().split()
      img['processed_tokens'].append(txt)
      if i < 10 and j == 0: 
        #import pdb; pdb.set_trace()
        txt_for_print = u''
        for attr in txt:
          txt_for_print += (attr.decode('utf-8') + u' ')
        sys.stdout.isatty(); print(txt_for_print); sys.stdout.flush()

def build_vocab(imgs, word_count_threshold=5):
  count_thr = word_count_threshold

  # count up the number of words
  counts = {}
  for img in imgs:
    for txt in img['processed_tokens']:
      for w in txt:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
  #print 'top words and their counts:'
  #print '\n'.join(map(str,cw[:20]))
  cw_for_print = []
  total_words = sum(counts.itervalues())
  #import pdb; pdb.set_trace()
  for count_word_tuple in cw:
    cw_for_print.append((count_word_tuple[0], count_word_tuple[1].decode('utf-8')))
  print 'top words and their counts:'
  total_percentage = 0.0
  for count_word_tuple in cw_for_print:
    percentage = count_word_tuple[0] * 1.0 / total_words
    total_percentage += percentage
    print(u'%d(%.5f): %s' % (count_word_tuple[0],  percentage * 100.0, count_word_tuple[1]))

  print('total percentage of words: %.5f' % (total_percentage * 100.0)); sys.stdout.flush()

  # print some stats
  #total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words(occurence <= %d): %d/%d = %.2f%%' % \
    (count_thr, len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
    for txt in img['processed_tokens']:
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  for i in xrange(max_len+1):
    print '%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
  for img in imgs:
    img['final_captions'] = []
    for txt in img['processed_tokens']:
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_captions'].append(caption)

  sys.stdout.flush()
  return vocab, max_len

def assign_splits(imgs, num_val = 4000, num_test = 0):
  num_val = num_val
  num_test =num_test

  for i,img in enumerate(imgs):
    if i < num_val:
      img['split'] = 'val'
    elif i < num_val + num_test: 
      img['split'] = 'test'
    else: 
      img['split'] = 'train'

  print 'assigned %d to val, %d to test.' % (num_val, num_test)
  sys.stdout.flush()

def encode_captions(imgs, wtoi, _max_length):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = _max_length
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print 'encoded captions to array of size ', `L.shape`
  sys.stdout.flush()
  return L, label_start_ix, label_end_ix, label_length

def main(params):

  filename = params['input_filename']
  #image_caption = [entry.strip().split(';;') for entry in open(filename, 'r')]
  image_caption = [entry.strip().split(',') for entry in open(filename, 'r')]
  # make reproducible
  seed(123)
  # shuffle the order
  shuffle(image_caption)

  imgs = []
  for item in image_caption:
    file_path = item[0]
    roi = item[1].split(' ')
    captions = []
    captions.append(item[2])
    imgs.append({'file_path': file_path, 'captions': captions, 'roi': roi})

  # tokenization and preprocessing
  prepro_captions(imgs)

  # create the vocab
  vocab, max_length = build_vocab(imgs, params['word_count_threshold'])
  # a 1-indexed vocab translation table
  itow = {i+1:w for i,w in enumerate(vocab)}
  # inverse table
  wtoi = {w:i+1 for i,w in enumerate(vocab)}

  # assign the splits
  assign_splits(imgs, params['num_val'], params['num_test'])
  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = \
    encode_captions(imgs, wtoi, max_length)

  # create output h5 file
  output_h5 = params['output_h5']
  image_dim = params['image_dim']
  N = len(imgs)
  f = h5py.File(output_h5, "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f.create_dataset("label_length", dtype='uint32', data=label_length)
  # space for resized images
  dset = f.create_dataset("images", (N,3,image_dim, image_dim), dtype='uint8')
  for i,img in enumerate(imgs):
    # load the image
    I = imread(img['file_path'])
    try:
      Ir = imresize(I, (image_dim, image_dim))
    except:
      print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
      raise
    # handle grayscale input images
    if len(Ir.shape) == 2:
      Ir = Ir[:,:,np.newaxis]
      Ir = np.concatenate((Ir,Ir,Ir), axis=2)
    # and swap order of axes from (params['image_dim'],params['image_dim'],3) to 
    # (3,params['image_dim'],params['image_dim'])
    Ir = Ir.transpose(2,0,1)
    # write to h5
    dset[i] = Ir
    if i % 1000 == 0:
      print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
      sys.stdout.flush()
  f.close()
  print 'wrote ', output_h5

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  for i,img in enumerate(imgs):
    
    jimg = {}
    jimg['split'] = img['split']
    if 'file_path' in img: jimg['file_path'] = img['file_path'] # copy it over, might need
    if 'id' in img: jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    
    out['images'].append(jimg)
  
  output_json = params['output_json']
  json.dump(out, open(output_json, 'w'))
  print 'wrote ', 'output_json'
  print 'end of building image-atrribute sentence'
  sys.stdout.flush()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--input_filename', default= \
    '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.shuffle.txt',
    #'/storage/freebee/tshirts_shirts_blous.image_sentence.txt',
    #'/storage/freebee/tshirts_shirts.image_sentence.txt',
    #'/storage/freebee/csv_backup/tshirts_excel_1453264869210.csv.image_sentence.txt',
    help='number of images to assign to validation data (for CV etc)')
  parser.add_argument('--num_val', default= \
    6400,
    #6000,
    #5000,
    #4000,
    type=int, 
    help='number of images to assign to validation data (for CV etc)')
  parser.add_argument('--num_test', default= \
    6400,
    #6000,
    #5000,
    #4000,
    type=int, 
    help='number of images to assign to tesst data (for CV etc)')
  parser.add_argument('--output_json', default= \
    '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.shuffle.txt.cutoff100.json', 
    #'/storage/freebee/tshirts_shirts_blous.image_sentence.txt.json', 
    #'/storage/freebee/tshirts_shirts.image_sentence.txt.json',
    #'/storage/freebee/tshirts.image_sentence.txt.json',
    help='output json file')
  parser.add_argument('--output_h5', default= \
    '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.shuffle.txt.cutoff100.h5', 
    #'/storage/freebee/tshirts_shirts_blous.image_sentence.txt.h5', 
    #'/storage/freebee/tshirts_shirts.image_sentence.txt.h5',
    #'/storage/freebee/tshirts.image_sentence.txt.h5',
    help='output h5 file')
  parser.add_argument('--word_count_threshold', default=100, type=int, 
    help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--image_dim', default=342, type=int, help='size of image')

  args = parser.parse_args()
  # convert to ordinary dict
  params = vars(args)
  print 'parsed input params:'
  print json.dumps(params, indent=2)
  main(params)

