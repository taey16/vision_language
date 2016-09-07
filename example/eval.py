#!/usr/bin/env python                                                                                      |  1
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

entries = [entry.strip().split('\t') for entry in \
            open(sys.argv[1], 'r')]

import pdb; pdb.set_trace()
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
  
  
