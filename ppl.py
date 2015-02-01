import bisect
import csv
import glob

import numpy
import scipy.stats

def NormalizeData(data):
  '''Normalize 'data' by rank-based inverse normal transformation.

  Args:
    data: a sequence of sequences containing raw scores.

  Returns:
    A list of lists matching the input but with values normalized.
  '''
  c = {}
  for scores in data:
    for sc in scores:
      c[sc] = c.get(sc, 0) + 1
  ks = c.keys()
  ks.sort()
  cum = 0
  d = {}
  s = numpy.sum(c.values())
  for k in ks:
    d[k] = scipy.stats.norm.ppf((cum + 0.5 * c[k]) / s)
    cum += c[k]
  return [[d[x] for x in y] for y in data]

def ForwardBackwardCurve(
    y_vals, p_o, sigmasqr_i, mu_0, ss_0, mu_o, ss_o, max_back):
  '''Dynamic programming algorithm similar to Forward-Backward to
  compute posterior outlier probabilities.'''
  n = y_vals.shape[0]
  ## the log-likelihood of the forward path ending at i not being outlier,
  ## the last one is just the sum of all paths
  forward_ll = numpy.zeros([n + 1])
  log_p_o = numpy.log(p_o)
  log_q_o = numpy.log(1.0 - p_o)
  oll = log_p_o-0.5*(y_vals - mu_o)**2/ss_o
  for i in xrange(n + 1):
    lls = []
    if i < max_back:
      lls.append(numpy.sum(oll[:i]) + log_q_o - 0.5*(y_vals[i] - mu_0)**2/ss_0)
    for j in xrange(min(i, max_back)):
      cur_val = numpy.sum(oll[i-j:i]) + forward_ll[i - j - 1]
      if (i < n):
        cur_val += log_q_o - 0.5*(y_vals[i] - y_vals[i - j - 1])**2/sigmasqr_i
      lls.append(cur_val)
    forward_ll[i] = sum_logs(lls)
  ## the log-likelihood of the forward path starting at i not being outlier
  backward_ll = numpy.zeros([n + 1])
  for i in xrange(n-1,-1,-1):
    lls = []
    for j in xrange(i+1, min(i+max_back+1,n+1)):
      cur_val = backward_ll[j]
      if j < n:
        cur_val += log_q_o - 0.5*(y_vals[j] - y_vals[i])**2/sigmasqr_i
      cur_val += numpy.sum(oll[i+1:j])
      lls.append(cur_val)
    backward_ll[i] = sum_logs(lls)
  sum_forward_backward = forward_ll + backward_ll
  posterior_q = numpy.exp(sum_forward_backward - forward_ll[n])
  return posterior_q[:-1]

def RemoveOutliers(data, min_plays=500):
  '''
  Removes outliers from 'data', a sequence of sequences, by computing
  posterior outlier probabilities according to a probabilistic model,
  then filtering out outlier points, removing curves that don't have
  at least 'min_plays' remaining non-outlier data points, and chopping
  all remaining curves down to the uniform length of 'min_plays'.
  '''
  x0s = numpy.array([x[0] for x in data])
  x0s_mean, x0s_var = (numpy.mean(x0s), numpy.var(x0s))
  res = []
  for plays in data:
    if len(plays) >= min_plays:
      y_vals = numpy.array(plays)
      posterior_q = ForwardBackwardCurve(
        y_vals, 0.05, 0.25, x0s_mean, x0s_var, -1.5, 1.0, 5)
      non_outlier_inds = [
        i for i in xrange(len(posterior_q)) if posterior_q[i] >= (1.0 / 4)]
      if len(non_outlier_inds) >= min_plays:
        res.append([plays[j] for j in non_outlier_inds[:min_plays]])
  return numpy.array(res)

def AffineCost(
    n, sumx, sumy, sumxy, sumx2, sumy2,
    with_ab=False, with_gh=False):
  'Cost function, gradient, hessian for curve fitting.'
  C1 = sumy2 - sumy**2/n
  C2a0 = sumxy - sumx*sumy/n
  C2a = C2a0**2
  C2b0 = sumx2 - sumx**2/n
  if C2b0 < 1e-6:
    if with_ab:
      return [0.0, sumy / n]
    if with_gh:
      return C1, numpy.zeros([3]), numpy.zeros([3, 3])
    return C1
  C2b = 1.0 / C2b0
  C2 = C2a * C2b
  C = C1 - C2
  if with_ab:
    z = 1.0 / (n * C2b0)
    a = z * n * C2a0
    b = z * (sumy * sumx2 - sumx*sumxy)
    return [a, b]
  if with_gh:
    gC2a0 = numpy.array([sumy * -1.0 / n, 1.0, 0.0])
    gC2a = 2 * C2a0 * gC2a0
    gC2b0 = numpy.array([-2.0 * sumx / n, 0.0, 1.0])
    gC2b = -gC2b0 / C2b0**2
    gC2 = C2a * gC2b + gC2a * C2b
    hC2a = 2 * gC2a0[:, numpy.newaxis] * gC2a0[numpy.newaxis, :]
    hC2b0 = numpy.zeros([3, 3])
    hC2b0[0, 0] = -2.0 / n
    hC2b = -(
      hC2b0 / C2b0**2 -
      2 * gC2b0[:, numpy.newaxis] * gC2b0[numpy.newaxis, :] / C2b0**3)
    hC2 = (
      gC2a[:, numpy.newaxis] * gC2b[numpy.newaxis, :] +
      gC2a[numpy.newaxis, :] * gC2b[:, numpy.newaxis] +
      C2a * hC2b +
      C2b * hC2a)
    gC = -gC2
    hC = -hC2
    return [C, gC, hC]
  return C

def AffineCost1(xs, ys, *args, **kwargs):
  'Wrapper for cost function for curve fitting.'
  n = len(xs)
  sumx = numpy.sum(xs)
  sumxy = numpy.sum(xs * ys)
  sumx2 = numpy.sum(xs**2)
  sumy = numpy.sum(ys)
  sumy2 = numpy.sum(ys**2)
  return AffineCost(n, sumx, sumy, sumxy, sumx2, sumy2, *args, **kwargs)

def AffineCost2(xs, ys, gxs, hxs):
  'Wrapper for cost, gradient, hessian for curve fitting.'
  g1 = numpy.array([
    numpy.sum(gxs, 1),
    numpy.sum(ys[numpy.newaxis, :] * gxs, 1),
    numpy.sum(2.0 * xs[numpy.newaxis, :] * gxs, 1)])
  h1 = numpy.array([
    numpy.sum(hxs, 2),
    numpy.sum(ys[numpy.newaxis, numpy.newaxis, :] * hxs, 2),
    numpy.sum(
      2.0 * gxs[:, numpy.newaxis, :] * gxs[numpy.newaxis, :, :] +
      2.0 * xs[numpy.newaxis, numpy.newaxis, :] * hxs, 2)])
  f, g, h = AffineCost1(xs, ys, with_gh=True)
  g2 = g.dot(g1)
  h2 = (g1.T.dot(h.dot(g1)) +
        numpy.sum(g[:, numpy.newaxis, numpy.newaxis] * h1, 0))
  return f, g2, h2

def MakeFGH(f, ys, xs=None):
  'Wrap a prediction function f on observed ys for inputs xs.'
  if xs is None:
    xs = numpy.arange(len(ys)) + 1.0
  just_f = lambda x: AffineCost1(f(xs, x), ys)
  def fgh(x):
    fxs, gxs, hxs = f(xs, x, with_gh=True)
    return AffineCost2(fxs, ys, gxs, hxs)
  return just_f, fgh

def ExpFGH(xs, a, with_gh=False):
  'Computes Exp function, gradient, hessian.'
  a2 = -numpy.exp(a[0])
  a2x = a2 * xs
  fx = numpy.exp(a2x)
  if with_gh:
    gx = a2x * fx
    hx = 2 * a2x * gx
    return fx, gx[numpy.newaxis, :], hx[numpy.newaxis, numpy.newaxis, :]
  return fx

def PLFGH(xs, a, with_gh=False):
  'Computes four-parameter PL function, gradient, hessian.'
  (c0, d0) = a
  c = -numpy.exp(c0)
  d = numpy.exp(d0)
  x1 = numpy.log(xs + d)
  x2 = c * x1
  x3 = numpy.exp(x2)
  if numpy.sum((x3 - numpy.mean(x3))**2) < 1e-15:
    if with_gh:
      return (numpy.zeros(xs.shape),
              numpy.zeros([2, xs.shape[0]]),
              numpy.zeros([2, 2, xs.shape[0]]))
    return numpy.zeros(xs.shape)
  if with_gh:
    gc = numpy.array([c, 0.0])
    gd = numpy.array([0.0, d])
    gx1 = gd[:, numpy.newaxis] / (xs[numpy.newaxis, :] + d)
    gx2 = gc[:, numpy.newaxis] * x1[numpy.newaxis, :] + c * gx1
    gx3 = x3 * gx2
    hc = numpy.zeros([2, 2])
    hc[0, 0] = c
    hd = numpy.zeros([2, 2])
    hd[1, 1] = d
    hx1 = (hd[:, :, numpy.newaxis] /
           (xs[numpy.newaxis, numpy.newaxis, :] + d) -
           (gd[:, numpy.newaxis, numpy.newaxis] *
            gd[numpy.newaxis, :, numpy.newaxis] /
            (xs[numpy.newaxis, numpy.newaxis, :] + d)**2))
    hx2 = (hc[:, :, numpy.newaxis] * x1 +
           gc[:, numpy.newaxis, numpy.newaxis] *
           gx1[numpy.newaxis, :, :] +
           gc[numpy.newaxis, :, numpy.newaxis] *
           gx1[:, numpy.newaxis, :] +
           c * hx1)
    hx3 = x3 * (hx2 + gx2[:, numpy.newaxis] * gx2[numpy.newaxis, :])
    return x3, gx3, hx3
  return x3

def WrapWithL2(f, fgh, l2_lmb):
  'Add a small L2 term for numerical stability.'
  f2 = lambda x: f(x) + 0.5 * l2_lmb * numpy.sum(x**2)
  def fgh2(x):
    fx, gx, hx = fgh(x)
    return (fx + 0.5 * l2_lmb * numpy.sum(x**2),
            gx + l2_lmb * x,
            hx + numpy.eye(len(x)) * l2_lmb)
  return f2, fgh2

def ComputeAllFits(model_f, x0, curve, min_len=50):
  'Compute the fits for a model on all windows of a curve.'
  l_curve = len(curve)
  n_prms = len(x0)
  errs_fits = numpy.zeros([l_curve, l_curve, n_prms + 4])
  errs_fits[:, :, 0] = numpy.inf
  l2_lmb = 1e-3
  xs0 = numpy.arange(l_curve + 1) + 1.0
  for start_i in xrange(l_curve):
    warm_start = numpy.array(x0)
    for l_fit0 in xrange(min_len - 1, l_curve - start_i):
      l_fit = l_fit0 + 1
      my_ys = curve[start_i:(start_i + l_fit)]
      f = lambda x: AffineCost1(model_f(xs0[:l_fit], x), my_ys)
      def fgh(x):
        xs, gxs, hxs = model_f(xs0[:l_fit], x, with_gh=True)
        return AffineCost2(xs, my_ys, gxs, hxs)
      f2, fgh2 = WrapWithL2(f, fgh, l2_lmb)
      sol = NewtonOptimize(f2, fgh2, warm_start)
      errs_fits[start_i, l_fit0, 0] = f(sol)
      sim_curve = model_f(xs0[:(1+l_fit)], sol)
      if len(sim_curve) + start_i > len(curve):
        sim_curve = sim_curve[:(len(curve)-start_i)]
      warm_start = sol
      ab = AffineCost1(sim_curve[:l_fit], my_ys, with_ab=True)
      errs_fits[start_i, l_fit0, 1] = (
        numpy.sum(
          (ab[1] + ab[0] * sim_curve[l_fit:] -
           curve[start_i+l_fit:start_i+l_fit+1])**2)
        if ((start_i + l_fit) < len(curve))
        else 0.0)
      full_params = numpy.concatenate([numpy.array(ab), sol])
      errs_fits[start_i, l_fit0, 2:] = full_params
  return errs_fits

## Some penalty functions

def BICPenalty(n, k):
  return k * (numpy.log(n) + numpy.log(2 * numpy.pi))

def AICPenalty(n, k):
  return k * 2.0

def AICcPenalty(n, k):
  return k * 2.0 + 2.0 * k * (k + 1.0) / (n - k - 1.0)

def PL1Penalty(n, k):
  return numpy.where(
    k < 5, numpy.zeros(k.shape), numpy.ones(k.shape) * numpy.inf)

def AccumulativePredictionError(
    errs_fits,
    min_len=50,
    max_parts=10,
    just_fit_all=False,
    v=None):
  '''Uses dynamic programming to compute APE for several models.
  M[i, j]: best fit with i+1 parts and length j+1
  M[0, j] = SSE[0, j]
  M[i, j] = min_k (M[i-1, k] + SSE[k, j-k])
  '''
  min_len = min_len
  l_curve = errs_fits.shape[0]
  n_params = errs_fits.shape[2] - 2
  M = numpy.zeros([max_parts, l_curve]) + numpy.inf
  B = numpy.zeros([max_parts, l_curve], dtype="int64")
  SSE = errs_fits[:, :, 0]
  M[0, :] = SSE[0, :]
  B[0, :] = 0
  SSE2 = numpy.zeros(SSE.shape)
  for j in xrange(l_curve):
    for k in xrange(j):
      SSE2[j, k] = SSE[k + 1, j - k - 1]
  for part_i in xrange(1, max_parts):
    for j in xrange(min_len - 1, l_curve):
      my_costs = M[part_i - 1, :j] + SSE2[j, :j]
      B[part_i, j] = numpy.argmin(my_costs)
      M[part_i, j] = my_costs[B[part_i, j]]
      if numpy.isnan(M[part_i, j]):
        print part_i, j
        return
  ics = [
    ("pl1", PL1Penalty),
    ("bic", BICPenalty),
    ("aic", AICPenalty),
    ("aicc", AICcPenalty)]
  n = (numpy.arange(l_curve) + 1.0)[numpy.newaxis, :]
  n_parts = (numpy.arange(max_parts) + 1.0)[:, numpy.newaxis]
  k = n_params * n_parts + (n_parts - 1)
  if v is None:
    ic0 = n * numpy.log(M + 1e-20)
  else:
    ic0 = M/v
  ape = {}
  for (ic_name, ic_func) in ics:
    penalty = ic0 + ic_func(n, k)
    ape[ic_name] = 0.0
    my_n_parts = numpy.argmin(penalty, 0)
    if just_fit_all:
      ape[ic_name] = my_n_parts[-1]
    else:
      for j in xrange(min_len, l_curve):
        start_i = B[my_n_parts[j], j]
        ape[ic_name] += errs_fits[start_i, j - start_i - 1, 1]
  return ape

def NewtonOptimize(
    f, fgh, x0,
    lmbsq_threshold = 1e-6,
    stepsize_threshold = 1e-5,
    beta = 0.5,
    sigma = 0.25):
  current_x = x0
  while True:
    fx, gx, hx = fgh(current_x)
    d = numpy.linalg.solve(hx, gx)
    lmbsq = numpy.sum(gx * d)
    if lmbsq < 0:
      d = -d
      lmbsq = -lmbsq
    if 0.5*lmbsq < lmbsq_threshold:
      break
    stepsize = 1.0
    while stepsize >= stepsize_threshold:
      new_x = current_x - stepsize * d
      new_fx = f(new_x)
      if (new_fx < (fx - sigma * stepsize * lmbsq)):
        current_x = new_x
        break
      stepsize *= beta
    if stepsize < stepsize_threshold:
      break
  return current_x

def TraceBestPath(errmat, min_cut=50):
  '''
  M is a (m,n+1) matrix where m is number of curves and n is total length.
  M[i,j] is the best path the ends at position j and curve i, not including j.
  So initialization is:
  M[0,j] = sum of errmat[0,k] for 0 <= k < j where j >= min_cut
  M[i,k] = infinity for k < min_cut
  Recursion:
  M[i,j] = min(M[i,j-1] + errmat[i,j-1], M[i-1,j-min_cut] + sum of errmat[i,k]
  for j-min_cut <= k < j.
  Best path is achieved by tracing back from M[m-1,n].
  '''
  m,n = errmat.shape
  M = numpy.ones([m,n+1]) * numpy.infty
  B = numpy.zeros([m,n+1], dtype='bool')
  M[0,min_cut:] = numpy.add.accumulate(errmat[0,:])[9:]
  for i in xrange(1, m):
    for j in xrange(min_cut, n+1):
      s1 = M[i,j-1] + errmat[i,j-1]
      s2 = M[i-1,j-min_cut] + numpy.sum(errmat[i,j-min_cut:j])
      if s1 > s2:
        M[i,j] = s2
        B[i,j] = True
      else:
        M[i,j] = s1
        B[i,j] = False
  sp = (m-1,n)
  pth = []
  while sp[0] > 0:
    if B[sp]:
      pth.append(sp[1] - min_cut)
      sp = (sp[0] - 1, sp[1] - min_cut)
    else:
      sp = (sp[0], sp[1] - 1)
  pth.append(0)
  pth.reverse()
  return pth, M[m-1,n]

def FindBestPaths(M, max_pl=10, min_cut=50):
  '''
  The error matrix has shape (max_pl, n+1):
  P[i,j] = the best error using i power laws and ending at j
  Hence the formula is: P[0,j] = M[0,j]
  P[i,j] = min_{0 <= k < j - min_cut} P[i-1,k] + M[k,j]
  '''
  n = M.shape[0]
  P = numpy.zeros([max_pl, n + 1]) + numpy.infty
  B = numpy.zeros([max_pl, n + 1], dtype='int32')
  P[0,:] = M[0,:]
  for i in xrange(1, max_pl):
    for j in xrange(1, n + 1):
      vals = P[i-1,:j] + M[:j,j]
      argm = numpy.argmin(vals)
      P[i,j] = vals[argm]
      B[i,j] = argm
  best_paths = [None] * max_pl
  for pl_i in xrange(max_pl):
    best_paths[pl_i] = [int(n)]
    j = pl_i
    while j > 0:
      best_paths[pl_i].append(B[j,best_paths[pl_i][-1]])
      j -= 1
    best_paths[pl_i].append(0)
    best_paths[pl_i] = map(int, best_paths[pl_i])
    best_paths[pl_i].reverse()
  return P, B, best_paths

def FitUpto(y_vals, max_n_pl, min_cut=50):
  'Fit all models up to max_n_pl pieces to y_vals.'
  y_vals = numpy.array(y_vals)
  M = ComputeAllFits(PLFGH, numpy.zeros([4]), y_vals, min_cut)[:, :, 1]
  P, B, best_paths = FindBestPaths(M, max_pl=max_n_pl, min_cut=min_cut)
  res = []
  for pl_i in xrange(max_n_pl):
    sc = 0.0
    bp_i = best_paths[pl_i]
    for j in xrange(1, len(bp_i)):
      sc += M[bp_i[j-1],bp_i[j]]
    res.append((best_paths[pl_i], sc))
  return res

def PrintICMetrics(sim_res):
  '''Print the metrics for the information criteria on simulation results.
  simres is a dictionary mapping from information criteria to tuples of
  three values: number of powerlaws, the real transition points, log of
  Akaike weights.
  '''
  res = []
  for ic in ['aic','aicc','bic']:
    simres = sim_res[ic]
    sc = 0.0
    sc2 = 0.0
    sc3 = 0.0
    sc4 = 0.0
    sc5 = 0.0
    sc6 = 0.0
    fp = 0.0
    fn = 0.0
    tp = 0.0
    tn = 0.0
    for (n_pls, real_cut_points, iclogs) in simres:
      assert (n_pls+1)==len(real_cut_points)
      iclogs2 = iclogs[:4]
      sc -= iclogs2[n_pls-1]
      lw = numpy.array(iclogs2)
      lw = lw - numpy.maximum.reduce(lw)
      w = numpy.exp(lw)
      w = w / numpy.sum(w)
      maxpred = numpy.argmax(w) + 1
      exppred = numpy.sum((numpy.arange(len(w)) + 1.0) * w)
      sc2 += (n_pls - maxpred)**2
      sc3 += (n_pls - exppred)**2
      sc4 += (maxpred != n_pls)
      if n_pls == 1:
        fp += 4.0 * int(maxpred > 1)
        tn += int(maxpred == 1)
      else:
        fn += 4.0 / 3 * int(maxpred == 1)
        tp += int(maxpred > 1)
      sc5 += (n_pls - maxpred)
      sc6 += (n_pls - exppred)
    res.append((ic,sc,sc2,sc3,sc4,fp,fn,sc5,sc6))
  print "%-3s %-7s %-7s %-7s"%("", "AIC", "AICc", "BIC")
  res = {x[0]: [y / 4000.0 for y in x[1:]] for x in res}
  method_names = ['LL', 'E1', 'E2', 'ER', 'FP', 'FN', 'B1', 'B2']
  for i in xrange(len(method_names)):
    print "%-3s %-7.4f %-7.4f %-7.4f"%(
      method_names[i], res['aic'][i], res['aicc'][i], res['bic'][i])

def MakeSimulatedCSV(fitted_curves):
  '''Make a CSV of simulated curves based on actual curves in fitted_curves.
  The fitted curves are used for the parameter values for the simulated
  curves - this is like a parametric bootstrap.'''
  pls = []
  howmany = 10000
  for (game_id, fit_data) in fitted_curves:
    for (k, plays, plays_dt, joint_curve, cut_points, pfu, plfits) in fit_data:
      if len(cut_points) == 2:
        pls.append(joint_curve)
        if len(pls) >= howmany:
          return pls
  curves_per_n_pls = {1: 1000, 2: 1000, 3: 1000, 4: 1000}
  noise_v = 0.096
  connection_v = 0.1
  cps = {}
  n_left = {n_pls: curves_per_n_pls.get(n_pls, 0) for n_pls in xrange(1,11)}
  ex_lambda = 1.0 / 0.005
  iter_i = 0
  min_cut = 50
  m = 500
  while sum(n_left.values()) > 0:
    iter_i += 1
    if (iter_i % 1000) == 0:
      print "%d iterations done."%(iter_i,)
    cut_points = [0]
    while cut_points[-1] < m:
      new_length = int(numpy.random.exponential() * ex_lambda + min_cut)
      cut_points.append(cut_points[-1] + new_length)
    cut_points[-1] = m
    if (cut_points[-1] - cut_points[-2]) < min_cut:
      continue
    k = len(cut_points) - 1
    if n_left[k] > 0:
      if not cps.has_key(k):
        cps[k] = []
      cps[k].append(cut_points)
      n_left[k] -= 1
  simdata = {}
  for k in cps.keys():
    simdata[k] = []
    for cut_points in cps[k]:
      joint_curve = []
      for j in xrange(1, len(cut_points)):
        pls_i = numpy.random.randint(0,len(pls))
        if j == 1:
          joint_curve.append(numpy.array(pls[pls_i][:cut_points[j]]))
        else:
          new_curve = pls[pls_i][cut_points[j-1]:cut_points[j]]
          delta = (
            joint_curve[-1][-1] - new_curve[0] +
            numpy.random.normal(size=[1])*numpy.sqrt(connection_v))
          joint_curve.append(numpy.array(new_curve) + delta)
      noise = numpy.random.normal(size=[cut_points[-1]]) * numpy.sqrt(noise_v)
      simdata[k].append(
        (cut_points, numpy.concatenate(joint_curve, 0) + noise))
  rows = []
  for k in xrange(1,5):
    for (cut_points, plays) in simdata[k]:
      unique_id = ''.join(
        ['%d'%(numpy.random.randint(0, 10),) for j in xrange(32)])
      row = (['%d'%(k,), '_'.join([unique_id]+map(repr, cut_points))] +
             map(str, plays) + ['0'] * 500)
      rows.append(row)
  csv.writer(open('simulated_curves.csv', 'wb')).writerows(rows)

def IterateSpaicc(fit_data, game_var=0.096):
  '''Iterate over the fit data to apply SPAICc on top of the fits.
  fit_data is an iterable of individual curve fits.
  game_var is the estimated noise for this game.
  '''
  for curve_data in fit_data:
    if simulated:
      (curvek, real_cut_points, v,
       cut_points, parts, v2, best_prob) = curve_data
      game_id = int(curvek.split(".")[0])
      user_id = curvek.split(".")[1]
      plays_dt = None
      curve_data = (game_id, user_id, v, plays_dt, cut_points,
                    parts, v2, best_prob)
    else:
      (game_id, user_id, v,
       plays_dt, cut_points, parts, v2, best_prob) = curve_data
    m = v.shape[0]
    k = len(parts)
    prv_curves = numpy.zeros([k, m])
    n_worst = 5
    worsts = numpy.zeros([k, n_worst], dtype='int32')
    err_matrix = numpy.zeros(prv_curves.shape)
    cum_err = numpy.zeros(prv_curves.shape)
    valid_pls = [True] * k
    for j in xrange(k):
      xs = numpy.arange(m - cut_points[j]) + 1.0
      prv_curves[j,cut_points[j]:] = powerlaw_f2(xs, *parts[j])
      err_matrix[j,cut_points[j]:] = (
        prv_curves[j,cut_points[j]:] - v[cut_points[j]:])**2
      cum_err[j,:] = numpy.add.accumulate(err_matrix[j,::-1])[::-1]
      worsts[j,:] = numpy.argsort(err_matrix[j,:])[-n_worst:]
      for j2 in xrange(j):
        s1 = cum_err[j2,cut_points[j]] - numpy.sum(err_matrix[j2,worsts[j2,:]])
        s2 = cum_err[j,cut_points[j]] - numpy.sum(err_matrix[j,worsts[j2,:]])
        if (s2/game_var + 10) >= (s1/game_var):
          valid_pls[j] = False
    new_cut_points = [0]
    new_parts = [parts[0]]
    for j in xrange(1, len(parts)):
      if valid_pls[j]:
        new_cut_points.append(cut_points[j])
        new_parts.append(parts[j])
    new_cut_points.append(v.shape[0])
    v3 = []
    for j in xrange(1, len(new_cut_points)):
      xs = numpy.arange(new_cut_points[j]-new_cut_points[j-1])+1.0
      v3.append(powerlaw_f2(xs, *new_parts[j-1]))
    v3 = numpy.concatenate(v3, 0)
    yield (game_id, user_id, v, plays_dt, new_cut_points,
           new_parts, v3, best_prob)

def EstimatedNoiseStatistics(curves):
  res = numpy.zeros([3])
  for v in curves:
    res[game_id] += numpy.sum(
      (v[2:] + v[:-2] -2 * v[1:-1])[numpy.newaxis, :]**
      numpy.arange(3)[:, numpy.newaxis], 1)
  return res

def LikelihoodRatioEstimatedNoise(estimated_noise):
  '''Computes the likelihood ratio test for zero mean noise.

  H0: x has zero mean, we estimate the variance
  H1: x has some mean and variance

  For H0 we set mu=0, sigma^2=s2/n.
  For H1 we set mu=s/n, sigma^2=s2/n-(s/n)^2.

  estimated_noise is a dictionary mapping from game id to the sufficient
  statistics (n, sum, sum of squares) of the estimated noise data for that
  game.
  '''
  en2 = dict(estimated_noise)
  en2["total"] = numpy.sum(en2.values(), 0)
  for gid in sorted(en2):
    (n, s, s2) = en2[gid]
    H0_params = (0.0, s2/n)
    H1_params = (s/n, s2/n-(s/n)**2)
    lls = [-0.5 * (
      n * numpy.log(2 * numpy.pi * sigma2) +
      (s2 + n * mu**2 - 2 * s * mu) / sigma2)
           for (mu, sigma2) in (H0_params, H1_params)]
    lldiff = lls[1] - lls[0]
    print gid, "chi2(1)=%.3f, p=%.3f"%(
      2.0 * lldiff, scipy.stats.chi2(1).sf(2.0 * lldiff))

def TestActualVarianceExplained(
    sigma2_lambda,
    n_curves,
    l_curve,
    a_mean, a_variance,
    b_mean, b_variance,
    c_mean, c_variance,
    geom_d,
    piece_length_lambda,
    min_piece_length):
  'Test the method for estimating variance explained.'
  sigma2 = numpy.random.exponential(sigma2_lambda)
  noise = numpy.random.normal(size=[n_curves, l_curve]) * numpy.sqrt(sigma2)
  ## Simulate data
  true_y = numpy.zeros([n_curves, l_curve])
  xs0 = numpy.arange(l_curve + 1) + 1.0
  for curve_i in xrange(n_curves):
    cut_points = [0]
    while cut_points[-1] < l_curve:
      new_cut_point = cut_points[-1] + int(numpy.random.exponential(
        piece_length_lambda)) + min_piece_length
      if new_cut_point > (l_curve - min_piece_length):
        new_cut_point = l_curve
      cut_points.append(new_cut_point)
    ## Simulate pieces
    last_x = 0.0
    for j in xrange(1, len(cut_points)):
      a = numpy.random.normal() * numpy.sqrt(a_variance) + a_mean
      b = numpy.random.normal() * numpy.sqrt(b_variance) + b_mean
      c = -numpy.exp(numpy.random.normal()*numpy.sqrt(c_variance)+c_mean)
      d = numpy.random.geometric(geom_d)
      piece_length = cut_points[j] - cut_points[j-1] + 1
      xs = xs0[:piece_length] + d
      piece_y = b + a*xs**c
      piece_y = piece_y[1:] - (piece_y[0] - last_x)
      true_y[curve_i, cut_points[j-1]:cut_points[j]] = piece_y
      last_x = piece_y[-1]
  xs0 = xs0[:l_curve]
  observed_y = true_y + noise
  est_noise = numpy.sum(
    (observed_y[:, 2:] + observed_y[:, :-2] - 2 * observed_y[:, 1:-1])**2)
  est_noise /= (6 * n_curves * (l_curve - 2.0))
  l2_lmb = 0.001
  SSE = 0.0
  real_SSE = 0.0
  for curve_i in xrange(n_curves):
    my_ys = observed_y[curve_i, :]
    f = lambda x: AffineCost1(ExpFGH(xs0, x), my_ys)
    def fgh(x):
      fxs, gxs, hxs = ExpFGH(xs0, x, with_gh=True)
      return AffineCost2(fxs, my_ys, gxs, hxs)
    f2, fgh2 = WrapWithL2(f, fgh, l2_lmb)
    sol = NewtonOptimize(f2, fgh2, -1.0 * numpy.ones([1]))
    my_xs = ExpFGH(xs0, sol)
    ab = AffineCost1(my_xs, my_ys, with_ab=True)
    pred = ab[0] * my_xs + ab[1]
    SSE += numpy.sum((pred - my_ys)**2)
    real_SSE += numpy.sum((pred - true_y[curve_i, :])**2)
  orig_var = numpy.sum(
    (true_y - numpy.mean(true_y, 1)[:, numpy.newaxis])**2)
  orig_var /= (n_curves * (l_curve - 1.0))
  est_MSE = SSE / (n_curves * (l_curve - 3.0))
  real_MSE = real_SSE / (n_curves * l_curve)
  observed_MSE = numpy.sum(
    (observed_y - numpy.mean(observed_y, 1)[:, numpy.newaxis])**2)
  observed_MSE /= (n_curves * (l_curve - 1.0))
  est_var_exp1 = 1.0 - est_MSE / observed_MSE
  est_var_exp2 = 1.0 - (est_MSE - est_noise) / (observed_MSE - est_noise)
  real_var_exp = 1.0 - real_MSE / orig_var
  return est_var_exp1, est_var_exp2, real_var_exp

def LikelihoodRatioFittingErrors(fiterr):
  '''
  Prints the likelihood-ratio test results for the fitting errors in fiterr.
  fiterr is a dictionary mapping game_ids to dictionaries mapping model names
  to a 4-tuple: (sum of squared errors, degrees of freedom, number of fit
  parameters, length of curve).
  '''
  for gid in sorted(fiterr):
    print "Game:", gid
    fg = {x: fiterr[gid][x][0] / fiterr[gid][x][1] for x in fiterr[gid]}
    tot = fg['variance']
    for model in sorted(fg):
      me = fg[model]
      print "%-20s: %.8f (%.8f)"%(model, me, (tot - me) / tot)
  ## Likelihood ratio test: 'pl1' is nested in 'sAICc'.
  for gid in sorted(fiterr):
    m1 = fiterr[gid]['pl1']
    m2 = fiterr[gid]['sAICc']
    lls = []
    ds = []
    ns = []
    for m in [m1, m2]:
      sse = m[0]
      n = m[3]
      v = sse / n
      ll = -0.5*(numpy.log(v) * n + sse / v)
      d = m[2]
      lls.append(ll)
      ds.append(d)
      ns.append(n / 500)
    lldiff = 2.0 * (lls[1] - lls[0])
    df = ds[1] - ds[0]
    ## Compute the log of the p-value, in base 10
    if df > 100:
      pval = scipy.stats.norm.logsf(
        (lldiff-df)/numpy.sqrt(2*df)) / numpy.log(10)
    else:
      pval = scipy.stats.chi2(df).logsf(lldiff) / numpy.log(10)
    print "Game %s, %d curves, Chi2(%.8f, %.8f), log10p = %.8f"%(
      gid, ns[0], lldiff, df, pval)
    ## Bonferroni correction is simply adding ns[0] because we have exactly
    ## 10 models per curve, and this is in log-10, so it should be
    ## ns[0] * log(10) / log(10) which is just ns[0]
    pval_bonf = pval + ns[0]
    print "After bonferroni: %.8f"%(pval_bonf,)

def CorrectedVarianceExplained(fiterr, data):
  '''Computes the estimated variance explained for the fitting errors for
  all models, fiterr. fiterr is a dictionary mapping game_ids to dictionaries
  mapping model names to a 4-tuple: (sum of squared errors, degrees of freedom,
  number of fit parameters, length of curve).
  data is a dictionary from game_id to an iterable of curves.
  It is used to estimated the measurement noise.
  '''
  rows = [["Task","AR", "1PL", "spAICc", "ppl3p",
           "pex3p", "delta1", "delta2", "delta3"]]
  enr = {}
  for game_id in data:
    enr[game_id] = numpy.zeros([3])
    for curve in data[game_id]:
      enr[game_id] += numpy.sum(
        (v[2:] + v[:-2] - 2 * v[1:-1])[numpy.newaxis, :]**
        numpy.arange(3)[:, numpy.newaxis], 1)
  enr0 = enr
  enr = {0: reduce(lambda x, y: x + y, enr.values())}
  enr.update(enr0)
  fiterr0 = fiterr
  fiterr = {0: fiterr['total']}
  fiterr.update(fiterr0)
  for game_id in [2, 5, 8, 10, 0]:
    basevar = fiterr[game_id]['variance'][0] / fiterr[game_id]['variance'][1]
    autoreg = (fiterr[game_id]['autoregressive'][0] /
               fiterr[game_id]['autoregressive'][1])
    pl1 = fiterr[game_id]['pl1'][0] / fiterr[game_id]['pl1'][1]
    spaicc = fiterr[game_id]['sAICc'][0] / fiterr[game_id]['sAICc'][1]
    ppl3p = fiterr[game_id]['ppl3p'][0] / fiterr[game_id]['ppl3p'][1]
    pex3p = fiterr[game_id]['pex3p'][0] / fiterr[game_id]['pex3p'][1]
    noise_est = enr[game_id][2] / enr[game_id][0] / 6
    noise_est = (enr[game_id][2] / enr[game_id][0] +
                 (enr[game_id][1] / enr[game_id][0])**2) / 6
    allvars = [basevar, noise_est, autoreg, pl1, spaicc]
    abs_row = ['abs',
               '%d'%(game_id,)] + ["%-.6f"%(x,) for x in allvars] + ["", ""]
    rel_row = ['rel',
               '%d'%(game_id,)] + ["%-.6f"%(x * 1.0 / basevar,)
                                   for x in allvars] + ["", ""]
    tru_var = basevar - noise_est
    ## deltas are the fraction of the remaining variance explained
    delta1 = (autoreg-pl1)/(autoreg-noise_est)
    delta2 = (pl1-spaicc)/(pl1-noise_est)
    delta3 = (spaicc-ppl3p)/(spaicc-noise_est)
    exp_row = ['exp',
               '%d'%(game_id,)] + [
                 "%-.6f"%(1.0 - (x - noise_est) / tru_var) for x in allvars]
    exp_row = exp_row + ["%-.6f"%(delta1,), "%-.6f"%(delta2,),
                         "%-.6f"%(delta3,)]
    exp_row = ["%d"%(game_id,)] + [
      "%-.2f"%(100 * (1.0 - (x - noise_est) / tru_var))
      for x in [autoreg, pl1, spaicc, ppl3p, pex3p]] + [
          "%-.2f"%(100 * x) for x in [delta1, delta2, delta3]]
    rows.append(exp_row)
  print print_table(rows)

def IterateWindowsAroundTransitions(
    fit_data, game_var, window_size=50, small_window_size=5, with_dt=True):
  ## compute background means by position
  ss = {}
  for curve_data in IterateSpaicc(fit_data, game_var):
    (game_id, user_id, v, plays_dt,
     cut_points, new_parts, v3, best_prob) = curve_data
    center_points = []
    for j in xrange(1, len(cut_points)):
      center_points.extend(
        range(cut_points[j-1]+window_size, cut_points[j]-window_size))
    for center_point in center_points:
      if center_point not in ss:
        ss[center_point] = numpy.zeros([3, window_size * 2 + 1])
      ss[center_point] += (
        v[center_point-window_size:center_point+window_size+1][
          numpy.newaxis,:]**numpy.arange(3)[:,numpy.newaxis])
  bmbp = {k: ss[k][1] / ss[k][0] for k in ss if ss[k][0][0] >= 10}
  for curve_data in IterateSpaicc(fit_data, game_var):
    (game_id, user_id, v, plays_dt,
     cut_points, new_parts, v3, best_prob) = curve_data
    for j in xrange(1, len(cut_points) - 1):
      if not with_dt:
        total_dt = None
      else:
        total_dt = (
          plays_dt[cut_points[j] + small_window_size] -
          plays_dt[cut_points[j] - small_window_size])
      my_window = (
        v[cut_points[j] - window_size:cut_points[j] + window_size + 1])
      if bmbp.has_key(cut_points[j]):
        my_window -= bmbp[cut_points[j]]
        yield curve_data + (j-1, my_window, total_dt)

def AverageTransitionsByRest(fit_data, game_var, with_time=True):
  itat = IterateWindowsAroundTransitions(fit_data, game_var)
  if with_time:
    splits = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.0, 1.0)]
  else:
    splits = [(0.0, 1.0)]
  all_t_rests = numpy.sort(numpy.array([x[-1] for x in itat]))
  curve_ss = [numpy.zeros([3, itat[0][-2].shape[0]]) for j in splits]
  for curve_data in itat:
    (game_id, user_id, v, plays_dt,
     cut_points, new_parts, v3, best_prob,
     pl_i, nrm_window, total_dt) = curve_data
    if with_time:
      qdt = bisect.bisect_right(all_t_rests, total_dt) * 1.0 / len(all_t_rests)
    else:
      qdt = 0.5
    for j in xrange(len(splits)):
      if (qdt >= splits[j][0]) and (qdt < splits[j][1]):
        curve_ss[j] += (
          nrm_window[numpy.newaxis, :] ** numpy.arange(3)[:, numpy.newaxis])
  mse = [None] * len(splits)
  for j in xrange(len(splits)):
    m = curve_ss[j][1] * 1.0 / curve_ss[j][0]
    v = curve_ss[j][2] * 1.0 / curve_ss[j][0] - m**2
    se = numpy.sqrt(v / curve_ss[j][0])
    m = m - numpy.mean(m)
    mse[j] = (m, se)
  return mse

def PL(x, u, a, c, d):
  n0 = numpy.exp(d)
  b = -numpy.exp(c)
  x1 = x + n0
  x2 = numpy.log(x1)
  x3 = b * x2
  x4 = numpy.exp(x3)
  x5 = a * x4
  x6 = u - x5
  return x6

def StepsForTransitionToImprove(fit_data, game_var):
  res = []
  for curve_data in IterateSpaicc(fit_data, game_var):
    (game_id, user_id, v, plays_dt,
     cut_points, parts, v_fit, best_prob) = curve_data
    for j in xrange(2, len(cut_points)):
      prv_extra = PL(
        numpy.arange(cut_points[j] - cut_points[j - 2]) + 1.0,
        *parts[j-2])[cut_points[j-1]-cut_points[j-2]:]
      cur_extra = PL(
        numpy.arange(cut_points[j] - cut_points[j - 1]) + 1.0, *parts[j-1])
      gt = numpy.nonzero(cur_extra > prv_extra)[0]
      res.append(None if len(gt) == 0 else gt[0])
  return res

def TransitionPointDistances(tps):
  '''Computes the histogram of distances between fit and true transition
  points.
  tps is an iterate of 2-tuples, a list of recovered transition
  points and the true cut points.
  '''
  h = {}
  for (rtp, cp) in tps:
    for j in cp:
      if len(rtp) == 0:
        d = -1
      else:
        di = numpy.argmin([abs(x-j) for x in rtp])
        myk = j-rtp[di]
        h[myk] = h.get(myk, 0) + 1
  return h

def AverageScoreDifferences(fitres, steps_points=[0, 25, 50, 75, 100]):
  '''Computes the average score differences between the transition moment
  and several time points later.
  fitres is a fit result iterator like the result of IterateSpaicc.'''
  res = {}
  n_steps = max(steps_points) + 1
  for curve_data in fitres:
    (game_id, user_id, v, plays_dt,
     cut_points, parts, v_fit, best_prob) = curve_data
    if game_id not in res:
      res[game_id] = []
    for j in xrange(1, len(parts)):
      prv_curve = PL(
        numpy.arange(n_steps) + 1.0 + (cut_points[j] - cut_points[j - 1]),
        *parts[j-1])
      cur_curve = PL(numpy.arange(n_steps) + 1.0, *parts[j])
      res[game_id].append([cur_curve[x] - prv_curve[x] for x in steps_points])
  return res
