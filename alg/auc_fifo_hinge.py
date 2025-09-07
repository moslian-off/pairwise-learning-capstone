import numpy as np
from sklearn import metrics
import timeit
import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils import get_idx

# 创建算法专用日志记录器
alg_logger = logging.getLogger("Algorithm")

def auc_fifo_hinge(x_tr, y_tr, x_te, y_te, options, w_t=None):
    """AUC FIFO Hinge算法"""
    # 获取轮次信息（如果提供）
    round_info = options.get('round_info', None)
    round_prefix = f"[轮次 {round_info}] " if round_info is not None else ""
    
    if w_t is None:
        w_t = np.zeros(options['dim'])
    
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    n_pass = options['n_pass']
    # 修复时间计算
    start = timeit.default_timer()
    ids = get_idx(n_tr, n_pass)
    n_iter = len(ids)
    stop = timeit.default_timer()
    time_avg = (stop - start) / n_iter if n_iter > 0 else 0
    res_idx = options['res_idx']
    if w_t is None:
        w_t = np.zeros(options['dim'])
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0
    eta_sum_old = 0
    eta = options['eta'] # initial eta
    etas = options['etas']
    beta = options['beta']
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    
    # 初始化loss记录
    losses = np.zeros(len(res_idx))
    
    # 打印总迭代次数
    message = f"{round_prefix}开始本地训练，总迭代次数: {n_iter}"
    print(message)
    alg_logger.info(message)
    
    # initiate timer
    start = timeit.default_timer()
    while t < n_iter:
        # current example
        y_t = y_tr[ids[t]] # only need to check first
        # past example
        y_t_1 = y_tr[ids[t-1]]

        # when ys are different
        if y_t * y_t_1 < 0:
            x_t = x_tr[ids[t]]
            x_t_1 = x_tr[ids[t - 1]]
            eta_t = etas[t]
            # make sure positive is in front
            yxx = y_t * (x_t - x_t_1)
            wyxx = np.inner(w_t, yxx)
            
            # 计算当前loss
            current_loss = max(0, 1 - wyxx)
            
            # 每10次迭代打印一次进度
            if t % 10 == 0:
                message = f"{round_prefix}迭代 {t}/{n_iter}: 当前loss = {current_loss:.6f}"
                print(message)
                alg_logger.info(message)
            
            # hinge loss gradient
            if 1 - wyxx > 0.:
                gd = - yxx
            else:
                gd = 0.
            # gradient step
            w_t = w_t - eta_t * gd
            if options['proj_flag']:
                # projection
                norm = np.linalg.norm(w_t)
                if norm > beta:
                    w_t = w_t * beta / norm
        
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
        
        # update
        t = t + 1
        
        # save results
        if i_res < len(res_idx) and res_idx[i_res] == t:
            # stop timer
            stop = timeit.default_timer()
            time_sum += stop - start + time_avg * (eta_sum - eta_sum_old)
            # average output
            w_avg = (w_sum - w_sum_old) / (eta_sum - eta_sum_old) # trick: only average between two i_res
            pred = (x_te.dot(w_avg.T)).ravel()
            if not np.all(np.isfinite(pred)):
                break
            
            # 计算AUC Score
            fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
            aucs[i_res] = metrics.auc(fpr, tpr)
            
            # 计算训练集上的loss
            train_pred = (x_tr.dot(w_avg.T)).ravel()
            train_loss = np.mean(np.maximum(0, 1 - y_tr * train_pred))
            losses[i_res] = train_loss
            
            times[i_res] = time_sum
            
            # 打印训练进度
            message = f"{round_prefix}评估点 {i_res + 1}/{len(res_idx)}: 迭代 {t}/{n_iter}, Loss = {train_loss:.6f}, AUC Score = {aucs[i_res]:.6f}, Time = {time_sum:.6f}s"
            alg_logger.info(message)
            
            i_res = i_res + 1
            w_sum_old = w_sum
            eta_sum_old = eta_sum
            # restart timer
            start = timeit.default_timer()
    
    # 打印训练完成信息
    message = f"{round_prefix}本地训练完成，总耗时: {time_sum:.6f}s"
    alg_logger.info(message)
    
    # 返回最终的参数、评估结果和loss记录
    return w_t, aucs, times, losses

if __name__ == '__main__':
    from common.utils import data_processing, get_res_idx
    from sklearn.model_selection import train_test_split

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 输出到标准输出，确保实时刷新
        ]
    )

    options = dict()
    options['n_pass'] = 10
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta'] = 0.1
    options['beta'] = 0.1
    options['eta_geo'] = 'const'
    options['proj_flag'] = True
    options['round_info'] = "测试"  # 添加轮次信息
    
    x, y = data_processing('ijcnn1')
    n, options['dim'] = x.shape
    options['n_tr'] = int(4 / 5 * n)
    options['etas'] = options['eta'] / np.sqrt(np.arange(1, options['n_pass'] * options['n_tr'] + 1))

    auc_sum = np.zeros(5)
    time_sum = np.zeros(5)
    loss_sum = np.zeros(5)
    for i in range(5):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=options['n_tr'])
        options['res_idx'] = get_res_idx(options['n_pass'] * (len(y_tr) - 1), options)

        w_final, aucs, times, losses = auc_fifo_hinge(x_tr, y_tr, x_te, y_te, options)
        auc_sum[i] = aucs[-1]
        time_sum[i] = times[-1]
        loss_sum[i] = losses[-1]