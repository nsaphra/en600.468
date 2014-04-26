import optparse
import sys
import bleu
from bisect import bisect_left
import copy
import random
from math import exp
import string
import math

grad = .05
MIN_W = -5.0
eps = .05

train_src_ = "data/train.src"
train_kbest_ = "data/train.100best"
train_ref_ = "data/train.ref"

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
    self.x[0] = self.x[1] - grad
    self.x[-1] = self.x[-2] + grad
    bleu_scores = [0.0 for x in self.x]
    for (ind, x) in enumerate(self.x[:-1]):
      bleu_scores[ind] = score_bleu_stats(self.y[ind])
      # if bleu_scores[ind] == 0:
      #   print self.y[ind]
    # stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(best.strip().split(),ref))]
    return max([((w+self.x[i+1])/2, score) for (i, (w, score)) in enumerate(zip(self.x[:-1], bleu_scores[:-1]))], key=lambda (x,y): y)

def get_score(slope, const, w):
  return slope * w + const

def line_intersect(const1, slope1, const2, slope2):
  w = (const1-const2)/(slope2-slope1)
  return (w, get_score(slope1, const1, w))

def sep_feat(feats, feat_name, weights):
  slope = feats[feat_name]
  const = sum([w * feats[name] if name != feat_name else 0.0 for (name, w) in weights.items()])
  return (slope, const)

def upper_envelope(kbest_h, kbest_feats, bleu_stats, feat_name, weights, ref___):
  data = sorted([(sep_feat(feats, feat_name, weights), h_ind, bleu_) for (h_ind, (feats, h, bleu_)) in enumerate(zip(kbest_feats, kbest_h, bleu_stats))],
    key=lambda ((prev_slope, prev_const), prev_h_ind, prev_bleu): prev_slope)
  # print data
  (prev_score, prev_sort_ind) = max([(get_score(slope, const, MIN_W), sort_ind) for (sort_ind, ((slope, const), h_ind, bleu_)) in enumerate(data)],
    key=lambda (x,y):x)
  ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[prev_sort_ind]
  ret = []
  prev_w_min = MIN_W
  while (prev_sort_ind < len(data)):
    (top_sort_ind, top_slope, top_const, top_w_min, top_score) = (-1, 0, 0, -MIN_W, -10000)
    for (sort_ind_add, ((slope, const), h_ind, curr_bleu)) in enumerate(data[prev_sort_ind:]):
      sort_ind = sort_ind_add+prev_sort_ind
      if slope == prev_slope:
        # curr_score = get_score(const, slope, top_w_min)
        # if curr_score > top_score:
        #   (top_sort_ind, top_slope, top_const, top_score) = (sort_ind, slope, const, curr_score)
        #   print "%d %d" % (h_ind, prev_h_ind)
        continue
      # print "ww0000000 %d %d" % (h_ind, prev_h_ind)
      assert(slope > prev_slope)
      (curr_w_min, curr_score) = line_intersect(const, slope, prev_const, prev_slope)
      if prev_w_min < curr_w_min and curr_w_min < top_w_min:
        (top_sort_ind, top_slope, top_const, top_w_min, top_score) = (sort_ind, slope, const, curr_w_min, curr_score)

    # print top_sort_ind
    if top_sort_ind < 0 or top_w_min >= -MIN_W:
      ret.append((prev_h_ind, prev_w_min, -MIN_W, prev_bleu, prev_score))
      break

    # TODO del
    # mid_w = (top_w_min+prev_w_min)/2
    # prev_top_score_ = get_score(prev_slope, prev_const, mid_w)
    # for (sort_i, ((slope_, const_), h_ind_, curr_bleu_)) in enumerate(data):
    #   score_ = get_score(slope_, const_, mid_w)
    #   next_score_ = get_score(slope_, const_, top_w_min+0.05)
    #   if sort_i == prev_sort_ind:
    #     if abs(prev_top_score_ - score_) > 1e-14:
    #       sys.stderr.write("%f %f\n" % (prev_top_score_, score_))
    #       assert(abs(prev_score - score_) < 1e-14)
    #   else:
    #     if score_ > prev_top_score_:
    #       print ("feat %s: curr w_min %f, curr w max %f" % (feat_name, prev_w_min, top_w_min))
    #       print ( "%f %f" % (prev_top_score_, score_))
    #   if sort_i != top_sort_ind:
    #     if score_ > top_score:
    #       print "%d %d %f %f\n" % (sort_i, prev_sort_ind, top_score, score_)
    #       assert False

        # if prev_top_score_ < prev_score:
        #   sys.stderr.write("feat %s: curr w_min %f, curr w max %f\n" % (feat_name, prev_w_min, top_w_min))
        #   sys.stderr.write( "%f %f\n" % (prev_top_score_, score_))
#          assert(False)

    # print prev_bleu
    ret.append((prev_h_ind, prev_w_min, top_w_min, prev_bleu, prev_score))
    prev_sort_ind = top_sort_ind
    prev_w_min = top_w_min
    prev_score = top_score
    ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[top_sort_ind]
  # print ret
  return ret

def mert(weights, data):
  # new_weights = copy.copy(weights)
  for name in weights.keys():
    print name
    f = piecewise_fcn(len(data))
    # (best_bleu, best_w) = (0.0, 0.0)
    # tmp_weights = copy.copy(new_weights)
    for (sent_ind, d) in enumerate(data):
      # print sent_ind
      for (h_ind, w_min, w_max, curr_bleu_stats, curr_score) in upper_envelope(d['kbest'], d['kbest_feats'], d['bleu'], name, weights, d['ref']):
        w_curr = ((w_min+w_max)/2)
        # tmp_weights[name] = w_curr
        
        # curr_bleu = performance(tmp_weights, train_src_, train_kbest_, train_ref_)
        # if curr_bleu > best_bleu:
        #   best_bleu = curr_bleu
        #   best_w = w_curr

        f.incr(w_min, w_max, sent_ind, curr_bleu_stats)
    (top_w, b) = f.find_max()
    # (top_w, b) = (best_w, best_bleu)
    sys.stderr.write("newbleu: %f\n" % b)
    # print "%s %f" % (name, top_w)
    weights[name] = top_w
    sys.stderr.write( "train BLEU %f\n" % performance(weights, train_src_, train_kbest_, train_ref_))
    sys.stderr.write( "test BLEU %f\n" % performance(weights, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref"))
  return weights

def get_feats(h, s, feats):
  f = {}
  for feat in feats.split(' '):
    (k, v) = feat.split('=')
    f[k] = float(v) # math.log(float(v)/10)
  f['word_cnt'] = 0.0
  f['untranslated_cnt'] = 0.0
  # f['word_cnt'] = exp((1.0 - len(s))/len(h)) if len(h) <= len(s) else 1.0
  # f['word_cnt'] = -(1.0 - len(s))/len(h) if len(h) <= len(s) else 0.0
  # f['untranslated_cnt'] = 0.0
  # for htok in h:
  #   if htok in s and htok not in string.punctuation:
  #     f['untranslated_cnt'] += 1.0
  # f['untranslated_cnt'] /= len(s)
  return f

def score_bleu_stats(bleu_stats):
  stats = [0.0 for i in xrange(10)]
  for s in bleu_stats:
    stats = [sum(scores) for scores in zip(s, stats)]
  return bleu.bleu(stats)

def performance(weights, dev_src, dev_kbest, dev_ref):
  old_weights = copy.deepcopy(weights)
  all_hyps = [pair.split(' ||| ') for pair in open(dev_kbest)]
  all_src = [s.split(' ||| ') for s in open(dev_src)]
  num_sents = len(all_hyps) / 100
  stats = []
  ref_file = open(dev_ref)

  for (r_ind, (ref, src, s)) in enumerate(zip(ref_file, all_src, xrange(0, num_sents))):
    hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    ref = ref.strip().split()
    src = src[1].strip().split()
    (best_score, best, best_ind) = (-1e300, '', -1)
    for (h_ind, (num, h_sent, feats)) in enumerate(hyps_for_one_sent):
      score = 0.0
      hyp = h_sent.strip().split()
      for (k,v) in get_feats(hyp, src, feats).items():
        score += weights[k] * v

      # if r_ind == 388:
      #   print "check %d %f" % (h_ind, score)

      f = get_feats(hyp, src, feats)
      for (k,v) in f.items():
        (slope, const) = sep_feat(f, k, weights)
        # if r_ind == 388:
        #   print "%s %f %f" % (k, slope, const)
        new_score = get_score(slope, const, weights[k])
        if abs(score - new_score) > 1e-13:
          print k
          print f
          print v
          print slope
          print const
          print slope*v + const
          print h_ind
          print "%f %f" % (score, new_score)
          assert(False)

      if score > best_score:
        (best_score, best, best_ind) = (score, h_sent, h_ind)
        # (best_score, best, best_ind, best_const) = (score, h_sent, h_ind, const)
    stats.append([i for i in bleu.bleu_stats(best.strip().split(),ref)])

  #   if r_ind == 388:
  #     f = get_feats(best.strip().split(), src, hyps_for_one_sent[best_ind][2])
  #     for (k,v) in f.items():
  #       (slope, const) = sep_feat(f, k, weights)
  #       print "%s %f %f" % (k, slope, const)
  #     print "%d %d %f" % (r_ind, best_ind, best_score)
  # print "====================="
  # assert(weights == old_weights)

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
  optparser.add_option("-c", "--word_cnt", dest="bp", default=0.5, type="float", help="brevity penalty weight")
  optparser.add_option("-g", "--greek_to_me", dest="untranslated", default=0.5, type="float", help="untranslated token weight")
  (opts, _) = optparser.parse_args()

  data = [{'ref':r.strip().split()} for r in open(opts.ref)]
  all_hyps = [pair.split(' ||| ') for pair in open(opts.kbest)]
  for (ind, s) in enumerate(open(opts.src)):
    (sent_id, src_sent) = s.split(' ||| ', 1)
    src = src_sent.strip().split()
    ref = data[ind]['ref']
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
  weights = {'p(e)' : opts.lm, 'p(e|f)' : opts.tm1, 'p_lex(f|e)' : opts.tm2, 'word_cnt':opts.bp, 'untranslated_cnt':opts.untranslated}
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
    old_weights = copy.deepcopy(weights)
    mert(weights, data)
    sys.stderr.write( "iter %d\n" % it)
    train_bleu = performance(weights, opts.src, opts.kbest, opts.ref)
    test_bleu = performance(weights, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref")
    sys.stderr.write( "train BLEU %f\n" % train_bleu)
    sys.stderr.write( "test BLEU %f\n" % test_bleu)
    out = ""
    for (n, w) in weights.items():
      out += "-%s %s " % (shortcuts[n], w)
    sys.stderr.write( out + "\n")
    if train_bleu > best_bleu:
      best_bleu = train_bleu
      best_test = test_bleu
      best_w = weights
    diff = 0.0
    for (n, w) in old_weights.items():
      diff += abs(w - weights[n])
    it += 1
    if diff <= eps or abs(train_bleu - prev_bleu) < eps:
      it += 1
      sys.stderr.write( "RANDOM RESTART\n")
      for name in weights.keys():
        weights[name] = random.uniform(MIN_W, -MIN_W)
  prev_bleu = train_bleu
  sys.stderr.write( "BEST:\n")
  sys.stderr.write( "train BLEU %f\n" % best_bleu)
  sys.stderr.write( "test BLEU %f\n" % performance(best_w, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref"))
  out = ""
  for (n, w) in best_w.items():
    out += "-%s %s " % (shortcuts[n], w)
  sys.stderr.write( out + "\n")

if __name__ == '__main__':
  main()