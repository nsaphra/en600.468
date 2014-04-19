import optparse
import sys
import bleu
from bisect import bisect_left

grad = .01
min_w = -1.0
eps = .01

class piecewise_fcn:
  def __init__(self):
    self.x = [min_w, -min_w]
    self.y = [0.0, 0.0]
    self.bleu_stats = 

  def find_x(self, x):
    ind = bisect_left(self.x, x)
    if ind < len(self.x)-1 and self.x[ind] == x:
      return ind
    if ind == 0 or ind == len(self.x):
      print "AGH WHAT"
      return -1
    self.x.insert(ind, x)
    self.y.insert(ind, self.y[ind-1])
    return ind

  def incr(self, min_x, max_x, added):
    min_ind = self.find_x(min_x)
    max_ind = self.find_x(max_x)
    for i in range(min_ind, max_ind):
      self.y[i] += added

  def find_max(self):
    return max(zip(self.x, self.y), key=lambda (x,y): y)


def line_intersect(c1, x1, c2, x2):
  x = (c1-c2)/(x2-x1)
  return (x, (c1 + x1*x))

def sep_feat(feats, feat_name, weights):
  slope = feats[feat_name]
  const = sum([w * feats[name] for (name, w) in weights.items()]) - slope*weights[feat_name]
  return (slope, const)


def upper_envelope(kbest_h, kbest_feats, bleu_scores, feat_name, weights):
  data = sorted([(sep_feat(feats, feat_name, weights), h, bleu_) for (feats, h, bleu_) in zip(kbest_feats, kbest_h, bleu_scores)])
  (curr_score, curr_ind) = max([(slope*min_w+const, ind) for (ind, ((slope, const), h, bleu_)) in enumerate(data)])
  ((prev_slope, prev_const), prev_h, prev_bleu) = data[curr_ind]
  ret = []
  prev_w_min = min_w
  while (curr_ind < len(kbest_h)):
    top_h = None
    (top_ind, top_slope, top_const, top_bleu, top_w_min, top_score) = (-1, 0, 0, 0, 0, -10000)
    for (ind, ((slope, const), h, curr_bleu)) in enumerate(data[curr_ind+1:]):
      if slope == prev_slope:
        continue
      (curr_w_min, curr_score) = line_intersect(const, slope, prev_const, prev_slope)
      # if prev_w_min > curr_w_min:
      #   print "AAAAAAH"
      if prev_w_min < curr_w_min and curr_w_min < top_w_min:
        top_ind = ind
        top_w_min = curr_w_min
        top_score = curr_score
    if top_ind < 0 or top_w_min >= -min_w:
      ret.append((prev_h, prev_w_min, -min_w, prev_bleu))
      break
    ret.append((prev_h, prev_w_min, top_w_min, prev_bleu))
    curr_ind = top_ind
    ((prev_slope, prev_const), prev_h, prev_bleu) = data[top_ind]
    prev_w_min = top_w_min
  return ret

def mert(weights, data):
  new_weights = {}
  out = ""
  shortcuts = {'p(e)' : 'l', 'p(e|f)' : 't', 'p_lex(f|e)' : 'u'}
  for (name, w) in weights.items():
    f = piecewise_fcn()
    for (ind, d) in data.items():
      (top_w_min, top_w_max, top_bleu) = (-1,-1,0)
      for (h, w_min, w_max, curr_bleu) in upper_envelope(d['kbest'], d['kbest_feats'], d['bleu'], name, weights):
        if curr_bleu > top_bleu:
          top_bleu = curr_bleu
          top_w_min = w_min
          top_w_max = w_max
      f.incr(top_w_min, top_w_max, top_bleu)
#      print "%f %f" % (top_w_min, top_w_max)
    (top_w, b) = f.find_max()
    new_weights[name] = top_w
    out += "-%s %s " % (shortcuts[name], w)
  print out
  return new_weights

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
  for (s, r) in zip(open(opts.src), open(opts.ref)):
    (ind, src_sent) = s.split(' ||| ', 1)
    data[ind] = {'src':src_sent.strip().split(), 'ref':r.strip().split(), 'kbest':[], 'kbest_feats':[], 'bleu':[]}
  for l in open(opts.kbest):
    (ind, h_sent, base_feats) = l.split(' ||| ')
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
      d['bleu'].append(bleu.bleu([sum(scores) for scores in zip(bleu_stats, bleu.bleu_stats(h,r))]))

  weights = {'p(e)' : opts.lm, 'p(e|f)' : opts.tm1, 'p_lex(f|e)' : opts.tm2} #, 'word_cnt':-1.0, 'untranslated_cnt':-1.0}
  for i in range(10):
    print "iter %d" % i
    new_weights = mert(weights, data)
    diff = 0.0
    for (n, w) in weights.items():
      diff += abs(w - new_weights[n])
    weights = new_weights
    if diff <= eps:
      break


if __name__ == '__main__':
  main()