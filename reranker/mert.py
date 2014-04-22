import optparse
import sys
import bleu
from bisect import bisect_left
import copy
import random
from math import exp
import string

grad = .05
MIN_W = -1.0
eps = .05

class piecewise_fcn:
  def __init__(self, num_sent):
    self.x = [MIN_W, -MIN_W]
    y_dflt = [-1 for sent in range(num_sent)]
    self.y = [y_dflt, copy.deepcopy(y_dflt)]

  def find_x(self, x, sent_ind):
    ind = bisect_left(self.x, x)
    if ind < len(self.x) and self.x[ind] == x:
      self.y[ind][sent_ind] = -1
      return ind
    self.x.insert(ind, x)
    self.y.insert(ind, copy.copy(self.y[ind-1]))
    self.y[ind][sent_ind] = -1
    return ind

  def incr(self, min_x, max_x, sent_ind, added):
    min_ind = self.find_x(min_x, sent_ind)
    max_ind = self.find_x(max_x, sent_ind)
    for i in range(min_ind, max_ind):
      self.y[i][sent_ind] = added
    if self.y[max_ind][sent_ind] == -1:
      self.y[max_ind][sent_ind] = added

  def find_max(self):
    bleu_scores = [0 for x in self.x]
    for (ind, x) in enumerate(self.x[:-1]):
      bleu_scores[ind] = score_bleu_stats(self.y[ind])
      # if bleu_scores[ind] == 0:
      #   print self.y[ind]
    # stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(best.strip().split(),ref))]
    return max([((w+self.x[i+1])/2, score) for (i, (w, score)) in enumerate(zip(self.x[:-1], bleu_scores[:-1]))], key=lambda (x,y): y)

def get_score(slope, const, w):
  return slope * w + const

def line_intersect(c1, x1, c2, x2):
  w = (c1-c2)/(x2-x1)
  return (w, get_score(x1, c1, w))

def sep_feat(feats, feat_name, weights):
  slope = feats[feat_name]
  const = sum([w * feats[name] if name != feat_name else 0 for (name, w) in weights.items()])
  return (slope, const)

def upper_envelope(kbest_h, kbest_feats, bleu_stats, feat_name, weights):
  data = sorted([(sep_feat(feats, feat_name, weights), h_ind, bleu_) for (h_ind, (feats, h, bleu_)) in enumerate(zip(kbest_feats, kbest_h, bleu_stats))], key=lambda ((prev_slope, prev_const), prev_h_ind, prev_bleu): prev_slope)
  (prev_score, prev_sort_ind) = max([(get_score(slope, const, MIN_W), sort_ind) for (sort_ind, ((slope, const), h_ind, bleu_)) in enumerate(data)], key=lambda (x,y):x)
  ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[prev_sort_ind]
  ret = []
  prev_w_min = MIN_W
  while (prev_sort_ind < len(data)):
    (top_sort_ind, top_slope, top_const, top_w_min, top_score) = (-1, 0, 0, -MIN_W, -10000)
    for (sort_ind_add, ((slope, const), h_ind, curr_bleu)) in enumerate(data[prev_sort_ind:]):
      sort_ind = sort_ind_add+prev_sort_ind
      if slope == prev_slope:
        continue
      assert(slope > prev_slope)
      (curr_w_min, curr_score) = line_intersect(const, slope, prev_const, prev_slope)
      if prev_w_min < curr_w_min and curr_w_min < top_w_min:
        top_sort_ind = sort_ind
        top_w_min = curr_w_min
        top_score = curr_score
        (top_sort_ind, top_slope, top_const, top_w_min, top_score) = (sort_ind, slope, const, curr_w_min, curr_score)

    if top_sort_ind < 0 or top_w_min >= -MIN_W:
      ret.append((prev_h_ind, prev_w_min, -MIN_W, prev_bleu))
      break

    # TODO del
    mid_w = (top_w_min+prev_w_min)/2
    prev_top_score_ = get_score(prev_slope, prev_const, mid_w)
    for (sort_i, ((slope_, const_), h_ind_, curr_bleu_)) in enumerate(data):
      score_ = get_score(slope_, const_, mid_w)
      if sort_i == prev_sort_ind:
        if abs(prev_top_score_ - score_) > 1e-14:
          sys.stderr.write("%f %f\n" % (prev_top_score_, score_))
          assert(abs(prev_score - score_) < 1e-14)
      else:
        if score_ > prev_top_score_:
          print ("feat %s: curr w_min %f, curr w max %f" % (feat_name, prev_w_min, top_w_min))
          print ( "%f %f" % (prev_top_score_, score_))
        # if prev_top_score_ < prev_score:
        #   sys.stderr.write("feat %s: curr w_min %f, curr w max %f\n" % (feat_name, prev_w_min, top_w_min))
        #   sys.stderr.write( "%f %f\n" % (prev_top_score_, score_))
#          assert(False)

    # print prev_bleu
    ret.append((prev_h_ind, prev_w_min, top_w_min, prev_bleu))
    prev_sort_ind = top_sort_ind
    prev_w_min = top_w_min
    prev_score = top_score
    ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[top_sort_ind]
  return ret

def mert(weights, data):
  new_weights = copy.copy(weights)
  for (name, w) in new_weights.items():
    f = piecewise_fcn(len(data))
    for (sent_ind, d) in enumerate(data):
      assert(len(d['kbest']) == len(d['kbest_feats']) and len(d['kbest_feats']) == len(d['bleu']))
      for (h_ind, w_min, w_max, curr_bleu_stats) in upper_envelope(d['kbest'], d['kbest_feats'], d['bleu'], name, new_weights):

        f.incr(w_min, w_max, sent_ind, curr_bleu_stats)
    (top_w, b) = f.find_max()
    sys.stderr.write("newbleu: %f\n" % b)
    new_weights[name] = top_w
  return new_weights

def get_feats(h, s, feats):
  f = {}
  for feat in feats.split(' '):
    (k, v) = feat.split('=')
    f[k] = float(v)
  # f['word_cnt'] = exp((1.0 - len(s))/len(h)) if len(h) <= len(s) else 1.0
  f['word_cnt'] = -(1.0 - len(s))/len(h) if len(h) <= len(s) else 0.0
  f['untranslated_cnt'] = 0.0
  for htok in h:
    if htok in s and htok not in string.punctuation:
      f['untranslated_cnt'] += 1.0
  # f['untranslated_cnt'] /= len(s)
  return f

def score_bleu_stats(bleu_stats):
  stats = [0 for i in xrange(10)]
  for s in bleu_stats:
    stats = [sum(scores) for scores in zip(s, stats)]
  return bleu.bleu(stats)

def performance(weights, dev_src, dev_kbest, dev_ref, print_out=False):
  all_hyps = [pair.split(' ||| ') for pair in open(dev_kbest)]
  num_sents = len(all_hyps) / 100
  stats = []
  ref_file = open(dev_ref)
  src_file = open(dev_src)
  for (ref, src, s) in zip(ref_file, src_file, xrange(0, num_sents)):
    hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    ref = ref.strip().split()
    src = src.strip().split()
    (best_score, best) = (-1e300, '')
    for (num, h_sent, feats) in hyps_for_one_sent:
      score = 0.0
      hyp = h_sent.strip().split()
      for (k,v) in get_feats(hyp, src, feats).items():
        score += weights[k] * v

      f = get_feats(hyp, src, feats)
      for (k,v) in f.items():
        (slope, const) = sep_feat(f, k, weights)
        new_score = get_score(slope, const, weights[k])
        # if abs(score - new_score) > 1e-13:
        #   print "%f %f" % (score, new_score)

      if score > best_score:
        (best_score, best) = (score, h_sent)
    stats.append([i for i in bleu.bleu_stats(best.strip().split(),ref)])
    # if print_out:
    #   print best
  return score_bleu_stats(stats)


# def performance_predef(weights, all_feats, bleu_stats):
#   stats = []
#   for (sent_feats, sent_bleu_stats) in zip(all_feats, bleu_stats):
#     (best_score, best) = (-1e300, [])
#     for (hyp_feats, hyp_bleu_stats) in zip(sent_feats, sent_bleu_stats):
#       score = 0.0
#       for (k, v) in hyp_feats.items():
#         score += weights[k] * v
#       if score > best_score:
#         (best_score, best) = (score, hyp_bleu_stats)
#     stats.append(best)
#   return score_bleu_stats(stats)


# def performance(weights, dev_src, dev_kbest, dev_ref, print_out=False):
#   all_hyps = [pair.split(' ||| ') for pair in open(dev_kbest)]
#   num_sents = len(all_hyps) / 100
#   ref_file = open(dev_ref)
#   src_file = open(dev_src)
#   all_feats = []
#   all_stats = []
#   for (ref, src, s) in zip(ref_file, src_file, xrange(0, num_sents)):
#     hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
#     ref = ref.strip().split()
#     src = src.strip().split()
#     sent_feats = []
#     sent_stats = []
#     for (it2, (num, h_sent, feats)) in enumerate(hyps_for_one_sent):
#       hyp = h_sent.strip().split()
#       sent_feats.append(get_feats(hyp, src, feats))
#       sent_stats.append([i for i in bleu.bleu_stats(hyp,ref)])
#     all_feats.append(sent_feats)
#     all_stats.append(sent_stats)
#   return performance_predef(weights, all_feats, all_stats)

def main():
  optparser = optparse.OptionParser()
  optparser.add_option("-s", "--src", dest="src", default="data/train.src", help="source sentences")
  optparser.add_option("-k", "--kbest-list", dest="kbest", default="data/train.100best", help="100-best translation lists")
  optparser.add_option("-r", "--ref", dest="ref", default="data/train.ref", help="reference translations")

  optparser.add_option("-l", "--lm", dest="lm", default=-0.5, type="float", help="Language model weight")
  optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
  optparser.add_option("-u", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
  (opts, _) = optparser.parse_args()

  data = [{'ref':r.strip().split()} for r in open(opts.ref)]
  all_hyps = [pair.split(' ||| ') for pair in open(opts.kbest)]
  for (ind, s) in enumerate(open(opts.src)):
    (sent_id, src_sent) = s.split(' ||| ', 1)
    src = src_sent.strip().split()
    ref = data[ind]['ref']
    ref_ = copy.copy(ref)
    data[ind]['src'] = src
    hyps_for_one_sent = all_hyps[ind * 100:ind * 100 + 100]
    data[ind]['kbest'] = [-1 for i in hyps_for_one_sent]
    data[ind]['kbest_feats'] = [-1 for i in hyps_for_one_sent]
    data[ind]['bleu'] = [-1 for i in hyps_for_one_sent]
    for (h_ind, (num, h_sent, feats)) in enumerate(hyps_for_one_sent):
      h = h_sent.strip().split()
      data[ind]['kbest'][h_ind] = h
      data[ind]['kbest_feats'][h_ind] = get_feats(h, src, feats)
      data[ind]['bleu'][h_ind] = [i for i in bleu.bleu_stats(h,ref)]

  shortcuts = {'p(e)' : 'l', 'p(e|f)' : 't', 'p_lex(f|e)' : 'u', 'word_cnt' : 'c', 'untranslated_cnt': 'g'}
  weights = {'p(e)' : opts.lm, 'p(e|f)' : opts.tm1, 'p_lex(f|e)' : opts.tm2, 'word_cnt':-0.5, 'untranslated_cnt':-0.5}
  sys.stderr.write( "iter -1\n")
  sys.stderr.write( "train BLEU %f\n" % performance(weights, opts.src, opts.kbest, opts.ref))
  sys.stderr.write( "test BLEU %f\n" % performance(weights, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref"))
  out = ""
  for (n, w) in weights.items():
    out += "-%s %s " % (shortcuts[n], w)
  sys.stderr.write( out + "\n")
  best_bleu = 0.0
  best_test = 0.0
  best_w = weights
  it = 0
  prev_bleu = 0.0
  while it < 5:
    new_weights = mert(weights, data)
    sys.stderr.write( "iter %d\n" % it)
    train_bleu = performance(new_weights, opts.src, opts.kbest, opts.ref)
    test_bleu = performance(new_weights, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref")
    sys.stderr.write( "train BLEU %f\n" % train_bleu)
    sys.stderr.write( "test BLEU %f\n" % test_bleu)
    out = ""
    for (n, w) in new_weights.items():
      out += "-%s %s " % (shortcuts[n], w)
    sys.stderr.write( out + "\n")
    if train_bleu > best_bleu:
      best_bleu = train_bleu
      best_test = test_bleu
      best_w = new_weights
    diff = 0.0
    for (n, w) in weights.items():
      diff += abs(w - new_weights[n])
    weights = new_weights
    if diff <= eps or abs(train_bleu - prev_bleu) < eps:
      it += 1
      break
      sys.stderr.write( "RANDOM RESTART\n")
      for name in weights.keys():
        weights[name] = random.uniform(MIN_W, -MIN_W)
  prev_bleu = train_bleu
  sys.stderr.write( "BEST:\n")
  sys.stderr.write( "train BLEU %f\n" % best_bleu)
  sys.stderr.write( "test BLEU %f\n" % performance(best_w, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref", print_out=True))
  out = ""
  for (n, w) in best_w.items():
    out += "-%s %s " % (shortcuts[n], w)
  sys.stderr.write( out + "\n")

if __name__ == '__main__':
  main()