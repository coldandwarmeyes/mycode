import numpy as np
import scipy


def regress_index_find_cross_subject(data, Q):
    [dim0, dim1] = np.shape(data)
    tmp1 = data[0, 0]
    [dim2, dim3] = np.shape(tmp1)
    tmp2 = tmp1[0, 0]
    dim4 = np.size(tmp2, 0) - Q
    dim_total = dim0 * dim1 * dim2 * dim3 * dim4
    idx = np.zeros((dim_total, 5))
    idx_idx = 0
    for i in range(dim0):
        for j in range(dim1):
            for u in range(dim2):
                for v in range(dim3):
                    for t in range(dim4):
                        idx[idx_idx, 0] = i
                        idx[idx_idx, 1] = j
                        idx[idx_idx, 2] = u
                        idx[idx_idx, 3] = v
                        idx[idx_idx, 4] = t+Q
                        idx_idx = idx_idx + 1
        idx_int=idx.astype(int)
    return idx_int

def regress_index_find_cross_day(data, Q, remove_head, remove_tail):
    [dim0, dim1] = np.shape(data)
    tmp1 = data[0, 0]
    dim2 = int(np.size(tmp1, 0) - Q - remove_head - remove_tail)
    dim_total = dim0 * dim1 * dim2
    idx = np.zeros((dim_total, 3))
    idx_idx = 0
    for i in range(dim0):
        for j in range(dim1):
            for u in range(dim2):
                idx[idx_idx, 0] = i
                idx[idx_idx, 1] = j
                idx[idx_idx, 2] = u+Q+remove_head
                idx_idx = idx_idx + 1
        idx_int=idx.astype(int)
    return idx_int


def myInterp2d(x,y,Nr,Nc,z,interp_kind):
    zn=np.zeros((Nr*2,Nc*2))
    for i in range(2):
        for j in range(2):
            z_tmp=z[i*len(x):(i+1)*len(x),j*len(y):(j+1)*len(y)]
            interpfunc = scipy.interpolate.interp2d(x, y, z_tmp, kind=interp_kind)
            xn = np.linspace(0, len(x)-1, num=Nr)
            yn = np.linspace(0, len(y)-1, num=Nc)
            zn_tmp = interpfunc(xn, yn)
            zn[i*Nr:(i+1)*Nr,j*Nc:(j+1)*Nc]=zn_tmp
    return zn



def my_rmse_ndof(y_pred, y_truth, idx):
    rmse=np.zeros((15,2,5))
    for i in range(15):
        for j in range(2):
            idx_trial=np.where( (idx[:, 0] == i) & (idx[:, 1] == j) )
            for k in range(5):
                y_pred_tmp=y_pred[idx_trial, k]
                y_truth_tmp=y_truth[idx_trial, k]
                rmse[i, j, k]=np.sqrt(np.mean(np.square(y_pred_tmp-y_truth_tmp)))
    return rmse

def my_rmse_1dof(y_pred, y_truth, idx):
    rmse=np.zeros((1,3))
    for j in range(3):
        idx_trial=np.where(idx[:, 1] == j)
        y_pred_tmp=y_pred[idx_trial]
        y_truth_tmp=y_truth[idx_trial]

        sum=0
        for k in range(len(y_truth_tmp)):
            sum=sum+pow((y_truth_tmp[k]-y_pred_tmp[k]),2)

        rmse[0, j]=np.sqrt(sum/len(y_truth_tmp))
    return rmse


