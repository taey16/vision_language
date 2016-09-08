#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

# NOTE:read prediction result
# format: predicted sequence of attributes\tground-truth sequence of attributes
entries = [entry.strip().split('\t') for entry in \
            open(sys.argv[1], 'r')]

# NOTE: compute prec/rec for each image, and then averaging them
#import pdb; pdb.set_trace()
average_precision = 0.0
average_recall = 0.0
for entry in entries:
  prediction = set(entry[0].strip().split(' '))
  gt = set(entry[1].strip().split(' '))
  hit_count = float(len(prediction.intersection(gt)))
  precision = hit_count / len(prediction)
  recall = hit_count / len(gt)

  average_precision += precision
  average_recall += recall

print 'average precision %f' % (average_precision / len(entries))
print 'average recall %f' % (average_recall / len(entries))
  
  
