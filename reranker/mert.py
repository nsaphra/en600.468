import optparse
import sys
import bleu
from bisect import bisect_left
import copy

grad = .01
min_w = -1.0
eps = .01

class piecewise_fcn:
  def __init__(self, num_sent):
    self.x = [min_w]
    y_dflt = [-1 for sent in range(num_sent)]
    self.y = [y_dflt]

  def find_x(self, x, sent_ind):
    ind = bisect_left(self.x, x)
    if ind < len(self.x)-1 and self.x[ind] == x:
      self.y[ind][sent_ind] = -1
      return ind
    if ind == 0 or ind > len(self.x):
      print "AGH WHAT"
      return -1
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
    for (ind, x) in enumerate(self.x):
      bleu_stats = [0 for i in range(10)]
      for s in self.y[ind]:
        bleu_stats = [sum(scores) for scores in zip(s, bleu_stats)]
      bleu_scores[ind] = bleu.bleu(bleu_stats)
    return max(zip(self.x, bleu_scores), key=lambda (x,y): y)


def line_intersect(c1, x1, c2, x2):
  x = (c1-c2)/(x2-x1)
  return (x, (c1 + x1*x))

def sep_feat(feats, feat_name, weights):
  slope = feats[feat_name]
  const = sum([w * feats[name] for (name, w) in weights.items()]) - slope*weights[feat_name]
  return (slope, const)


def upper_envelope(kbest_h, kbest_feats, bleu_stats, feat_name, weights):
  data = sorted([(sep_feat(feats, feat_name, weights), h_ind, bleu_) for (h_ind, (feats, h, bleu_)) in enumerate(zip(kbest_feats, kbest_h, bleu_stats))], key=lambda ((prev_slope, prev_const), prev_h_ind, prev_bleu): prev_slope)
  (prev_score, prev_sort_ind) = max([(slope*min_w+const, sort_ind) for (sort_ind, ((slope, const), h_ind, bleu_)) in enumerate(data)], key=lambda (x,y):x)
  ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[prev_sort_ind]
  ret = []
  prev_w_min = min_w
  while (prev_sort_ind < len(kbest_h)):
    (top_sort_ind, top_slope, top_const, top_w_min, top_score) = (-1, 0, 0, -min_w, -10000)
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

    if top_sort_ind < 0 or top_w_min >= -min_w:
      ret.append((prev_h_ind, prev_w_min, -min_w, prev_bleu))
      break
    ret.append((prev_h_ind, prev_w_min, top_w_min, prev_bleu))
    prev_sort_ind = top_sort_ind
    prev_w_min = top_w_min
    ((prev_slope, prev_const), prev_h_ind, prev_bleu) = data[top_sort_ind]

    # assert(curr_w_min > prev_w_min)
    top_h_ind = data[top_sort_ind][1]
    # score = 0.0
    # for (k, feat) in kbest_feats[top_h_ind].items():
    #   if k == feat_name:
    #     score += feat*top_w_min
    #   else:
    #     score += feat * weights[k]
    # score__ = sum([w * kbest_feats[top_h_ind][name] for (name, w) in weights.items()]) - kbest_feats[top_h_ind][feat_name] * weights[feat_name] + kbest_feats[top_h_ind][feat_name] * top_w_min
    # (s_, c_) = sep_feat(kbest_feats[top_h_ind], feat_name, weights)
    # print "====="
    # print "%f == %f == %f == %f" % (top_score, score__, score, c_+top_w_min*kbest_feats[top_h_ind][feat_name])
    # print top_score - score__
    # print top_score - score
    # assert(top_score == score__)
    # assert(top_score == score)
    # for (i, feats) in enumerate(kbest_feats):
    #   score= sum([w * kbest_feats[i][name] for (name, w) in weights.items()]) - kbest_feats[i][feat_name] * weights[feat_name] + kbest_feats[i][feat_name] * top_w_min
    #   if i == top_h_ind:
    #     assert(top_score == score)
    #   else:
    #     if top_score < score:
    #       print "%f %f %f" % (slope, const, slope*top_w_min+const)
    #       print top_score - score
    #       print "%d %d" % (i, top_h_ind)
    #       print "%f < %f" % (top_score, score)
        #assert(top_score >= score)

  return ret

def mert(weights, data):
  new_weights = {}
  out = ""
  shortcuts = {'p(e)' : 'l', 'p(e|f)' : 't', 'p_lex(f|e)' : 'u'}
  for (name, w) in weights.items():
    f = piecewise_fcn(len(data))
    for (sent_ind, d) in data.items():
      for (h_ind, w_min, w_max, curr_bleu_stats) in upper_envelope(d['kbest'], d['kbest_feats'], d['bleu'], name, weights):
        f.incr(w_min, w_max, sent_ind, curr_bleu_stats)
    for b in f.y:
      if -1 in b:
        print b
    (top_w, b) = f.find_max()
    new_weights[name] = top_w
    out += "-%s %s " % (shortcuts[name], w)
  print out
  return new_weights

def performance(weights, dev_src, dev_kbest, dev_ref):
  all_hyps = [pair.split(' ||| ') for pair in open(dev_kbest)]
  num_sents = len(all_hyps) / 100
  stats = [0 for i in xrange(10)]
  ref_file = open(dev_ref)
  for (ref, s) in zip(ref_file, xrange(0, num_sents)):
    hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    ref = ref.strip().split()
    (best_score, best) = (-1e300, '')
    for (num, hyp, feats) in hyps_for_one_sent:
      score = 0.0
      for feat in feats.split(' '):
        (k, v) = feat.split('=')
        score += weights[k] * float(v)
      if score > best_score:
        (best_score, best) = (score, hyp)
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(best.strip().split(),ref))]
  return bleu.bleu(stats)

def main():
  optparser = optparse.OptionParser()
  optparser.add_option("-s", "--src", dest="src", default="data/train.src", help="source sentences")
  optparser.add_option("-k", "--kbest-list", dest="kbest", default="data/train.100best", help="100-best translation lists")
  optparser.add_option("-r", "--ref", dest="ref", default="data/train.ref", help="reference translations")

  optparser.add_option("-l", "--lm", dest="lm", default=-0.5, type="float", help="Language model weight")
  optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
  optparser.add_option("-u", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
  (opts, _) = optparser.parse_args()


  data = {}
  sent_ind_from_id = {}
  for (ind, (s, r)) in enumerate(zip(open(opts.src), open(opts.ref))):
    (sent_id, src_sent) = s.split(' ||| ', 1)
    sent_ind_from_id[sent_id] = ind
    data[ind] = {'src':src_sent.strip().split(), 'ref':r.strip().split(), 'kbest':[], 'kbest_feats':[], 'bleu':[]}
  for l in open(opts.kbest):
    (sent_id, h_sent, base_feats) = l.split(' ||| ')
    ind = sent_ind_from_id[sent_id]
    s = data[ind]['src']
    h = h_sent.strip().split()
    feats = {}
    for feat in base_feats.split(' '):
      (k, v) = feat.split('=')
      feats[k] = float(v)
    # feats['word_cnt'] = len(h)
    # feats['untranslated_cnt'] = 0
    # for htok in h:
    #   if htok in s:
    #     feats['untranslated_cnt'] += 1
    data[ind]['kbest'].append(h)
    data[ind]['kbest_feats'].append(feats)

  bleu_stats = [0 for i in range(10)]
  for (ind, d) in data.items():
    bleu_stats = [sum(scores) for scores in zip(bleu_stats, bleu.bleu_stats(d['ref'], d['ref']))]
  for (ind, d) in data.items():
    r = d['ref']
    for (h, feats) in zip(d['kbest'], d['kbest_feats']):
      d['bleu'].append([score for score in bleu.bleu_stats(h,r)])

  weights = {'p(e)' : opts.lm, 'p(e|f)' : opts.tm1, 'p_lex(f|e)' : opts.tm2} #, 'word_cnt':-1.0, 'untranslated_cnt':-1.0}
  for i in range(10):
    print "iter %d" % i
    print "train BLEU %f" % performance(weights, opts.src, opts.kbest, opts.ref)
    print "test BLEU %f" % performance(weights, "data/dev+test.src", "data/dev+test.100best", "data/dev.ref")
    new_weights = mert(weights, data)
    diff = 0.0
    for (n, w) in weights.items():
      diff += abs(w - new_weights[n])
    weights = new_weights
    if diff <= eps:
      break



if __name__ == '__main__':
  main()