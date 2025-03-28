import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq as sp_lstsq
from scipy.linalg import circulant
from scipy.optimize import nnls
import copy
import numba as nb
from mindxlib.visualization.plots import plot_static_gam
from mindxlib.visualization.interactive import create_app

@nb.jit(nopython=True)
def ReLu_product(x,w,split,index_list):
    result = np.zeros(x.shape[0])
    for idx, sp in enumerate(split):
        result[index_list[idx]:] += w[idx]*(x[index_list[idx]:]-sp)
    return result

class shapeFunctionOptimizer:
    def __init__(self,block_size=50,lambda_1=0.1,eta=0.001,step_size=1,bin_num=64,\
        momentum=0.9,momentum_type='Huber',reg_type='Huber',randomize=False):
        self.block_size = block_size
        self.lambda_1 = lambda_1/self.block_size
        self.eta = eta
        self.bin_num = bin_num
        self.active_threshold = None
        self.active_index = None
        self.step_size = step_size
        self.relu_weight = None
        self.constant = 0
        self.sorted_x = None
        self.sorted_index = None
        self.Q = None
        self.q = None
        self.constraints_controller = constraints_controller()
        self.momentum = momentum
        self.momentum_type = momentum_type
        self.reg_type = reg_type
        self.randomize = randomize
        self.eps = 1

    def _init_model(self,x,sample_weight):
        sorted_x = np.sort(x)
        sorted_index = np.argsort(x)
        self.sorted_x = sorted_x
        self.sorted_index = sorted_index
        self.lambda_1 = self.lambda_1 * np.sum(sample_weight.flatten())
        self.sample_weight = sample_weight[self.sorted_index]
        self.threshold_list, self.threshold_index = self._getSplitList(self.sorted_x,self.bin_num)
        self.N = len(self.threshold_list)
        self.block_size = min(self.block_size,self.N)
        self.active_threshold = np.zeros(self.block_size)
        self.active_index = np.zeros(self.block_size,dtype='int32')
        self.relu_weight = np.zeros(self.N)
        self.Q = np.zeros((self.block_size+1,self.block_size+1))
        self.q = np.zeros(self.block_size+1)
        self.x_re = self.sorted_x[::-1]
        sample_weight_re = self.sample_weight[::-1]
        self.wx2sum = np.cumsum((self.x_re**2)*sample_weight_re)[::-1]
        self.wxsum = np.cumsum(self.x_re*sample_weight_re)[::-1]
        self.wsum = np.cumsum(sample_weight_re)[::-1]
        A = (np.ones((self.block_size,1)).dot(np.arange(self.block_size).reshape(1,-1))).astype('int')
        self.index_m = np.maximum(A,A.T)
        self.Q[0,0] = self.wsum[0]
    
    def _getSplitList(self,x,bin_num):
        threshold_list, threshold_index = np.unique(x,return_index=True)
        if (bin_num is None) or (threshold_list.shape[0] < bin_num):
            bin_num = threshold_list.shape[0]
        PP = threshold_list.shape[0]/bin_num
        pp_indices = np.round(np.arange(bin_num)*PP).astype('int32')
        return threshold_list[pp_indices], threshold_index[pp_indices]

    
    def _generate_Q(self,active_threshold,active_index):
        # sample_weight_re = self.sample_weight[::-1]
        # self.wx2sum = np.cumsum((self.x_re**2)*sample_weight_re)[::-1]
        # self.wxsum = np.cumsum(self.x_re*sample_weight_re)[::-1]
        # self.wsum = np.cumsum(sample_weight_re)[::-1]
        B = np.take(active_index,self.index_m)
        a = active_threshold.reshape(-1,1)
        Q_1 =  a.dot(a.T)*np.take(self.wsum,B) - (a+a.T)*np.take(self.wxsum,B) + np.take(self.wx2sum,B)
        self.Q[1:,1:] = Q_1
        temp_v = np.take(self.wxsum,active_index) - np.take(self.wsum,active_index)*active_threshold
        self.Q[0,1:] = temp_v
        self.Q[1:,0] = temp_v
    
    def _estimate_grad(self,y):
        wy = self.sample_weight*y
        wxy = self.sorted_x*wy
        wxysum = np.cumsum(wxy[::-1])[::-1]
        wysum = np.cumsum(wy[::-1])[::-1]
        grad = wxysum[self.threshold_index] - wysum[self.threshold_index] * self.threshold_list
        return grad


    def _generate_q(self,y,active_threshold,active_index):
        if self.randomize:
            wy = (1*(np.random.rand(y.shape[0]))+0.5)*self.sample_weight*y
        else:
            wy = self.sample_weight*y
        wxy = self.sorted_x*wy
        wxysum = np.cumsum(wxy[::-1])[::-1]
        wysum = np.cumsum(wy[::-1])[::-1]
        self.q[1:]=np.take(wxysum,active_index) - np.take(wysum,active_index)*active_threshold
        self.q[0] = wysum[0]

    def step(self, y, Regular=True):
        # grad = abs(self.lambda_1*(1-self.momentum) - self._estimate_grad(y))
        # self.active_index[1:] = np.sort(np.argpartition(grad[1:],-self.block_size+1)[-self.block_size+1:]) + 1

        self.active_index[1:] = np.sort(np.random.choice(self.N-1,self.block_size-1,replace=False)) + 1
        self.active_threshold = self.threshold_list[self.active_index]
        active_index_on_x = self.threshold_index[self.active_index]
        current_weight = self.relu_weight[self.active_index]
        if Regular:
            if (self.momentum_type == 'Huber') and (self.reg_type == 'Huber'):
                Rege_vector = 1/(self.eps+self.eta+np.abs(current_weight)**2)
                Hessian_vector = Rege_vector
                # Rege_vector = np.insert(1/(self.eta+np.abs(self.relu_weight[self.active_index])),0,0)
                # Hessian_vector = Rege_vector
            elif (self.momentum_type == 'L2') and (self.reg_type == 'Huber'):
                Rege_vector = 1/(self.eps+self.eta+np.abs(self.relu_weight[self.active_index]))
                Hessian_vector = self.momentum + (1-self.momentum)*Rege_vector
            elif (self.momentum_type == 'Huber') and (self.reg_type == 'L2'):
                Rege_vector = np.ones(self.block_size)
                Hessian_vector = self.momentum/(self.eps+self.eta+np.abs(current_weight)) + (1-self.momentum)*Rege_vector
            else:
                Rege_vector = np.ones(self.block_size)
                Hessian_vector = Rege_vector
            Rege_vector[0] = 0 # Do not add regularization on the first ReLU
            Hessian_vector[0] = 0
            Rege_vector = np.insert(Rege_vector,0,0) # insert the term of constant term
            Hessian_vector = np.insert(Hessian_vector,0,0)
            Hessian_matrix = np.diag(Hessian_vector)
            Hessian_matrix[0,0] = 0
        else:
            Hessian_matrix = 0
            Rege_vector = 0
        self._generate_Q(active_threshold=self.active_threshold,active_index=active_index_on_x)
        self._generate_q(y[self.sorted_index],active_threshold=self.active_threshold,active_index=active_index_on_x)
        Q = self.Q + self.lambda_1*Hessian_matrix
        q = self.q - (1-self.momentum)*self.lambda_1*Rege_vector*np.insert(self.relu_weight[self.active_index],0,0)
        w = sp_lstsq(Q, q, lapack_driver='gelsy', check_finite=False)[0]
        w[1:] = self.constraints_controller.project_on_constrains(self.relu_weight, w[1:],self.active_index)
        self.relu_weight[self.active_index] += self.step_size*w[1:]
        self.constant += self.step_size*w[0]
        piecewise_app = ReLu_product(self.sorted_x,w[1:],self.active_threshold,active_index_on_x) + w[0]
        update_v = np.zeros(self.sorted_x.shape)
        update_v[self.sorted_index] = piecewise_app
        self.eps = self.eps*0.95
        return update_v
    
    def prepare_prediction(self):
        slope = np.cumsum(self.relu_weight)
        shape_func = np.cumsum(np.insert(slope[:-1]*np.diff(self.threshold_list),0,self.constant))
        self.refer_points = np.insert(self.threshold_list,0,self.threshold_list[0])
        self.slope = np.insert(slope,0,slope[0])
        self.ref_func_v = np.insert(shape_func,0,shape_func[0])
    
    def predict(self,x):
        index = np.searchsorted(self.threshold_list,x)
        return self.ref_func_v[index] + self.slope[index]*(x-self.refer_points[index])
    
    def get_para(self):
        values = self.predict(self.sorted_x)
        return self.sorted_x, values
    
    def add_constraints(self,cons_list):
        for cons in cons_list:
            self.constraints_controller.add_constraints(cons['left'],cons['right'],cons['type'])
            
        self.relu_weight = self.constraints_controller.analysis_constraints(self.threshold_list,self.relu_weight)

    

class constraints_controller:
    def __init__(self):
        self.first_order_constraints_list = []
        self.second_order_constraints_list = []
        
    def add_constraints(self,left,right,const_type):
        if left >= right:
            return
        if const_type not in ['Increase', 'Decrease', 'Convex', 'Concave']:
            return
        
        constraint = {}
        constraint['type'] = const_type
        constraint['left'] = left
        constraint['right'] = right

        if (const_type == 'Increase') or (const_type == 'Decrease'):
            self.first_order_constraints_list.append(constraint)
        else:
            self.second_order_constraints_list.append(constraint)
    
    def _merge_constraints(self,constraints_list,confict_type):
        constraints_list = sorted(constraints_list,key=lambda x:x['left_index'])
        merged_constraint = []
        # for constraint in constraints_list:
        #     if (constraint['type'] == 'Increase') or (constraint['type'] == 'Covex'):
        #         constraint['sign'] = 1
        #     else:
        #         constraint['sign'] = -1
        current_constraint = None
        for constraint in constraints_list:
            if current_constraint is None:
                current_constraint = constraint
                continue

            if constraint['left_index'] > current_constraint['right_index']:
                merged_constraint.append(current_constraint)
                current_constraint = constraint
            else:
                if constraint['type'] == current_constraint['type']:
                    current_constraint['right'] = max(current_constraint['right'],constraint['right'])
                    current_constraint['right_index'] = max(current_constraint['right_index'],constraint['right_index'])
                elif constraint['left_index'] == current_constraint['right_index']: # 连续约束的情况
                    merged_constraint.append(current_constraint)
                    current_constraint = constraint
                else: # 约束冲突
                    cons = {} # 构建冲突部分约束
                    cons['type'] = confict_type
                    cons['left'] = constraint['left']
                    cons['right'] = min(constraint['right'],current_constraint['right'])
                    cons['left_index'] = constraint['left_index']
                    cons['right_index'] = min(constraint['right_index'],current_constraint['right_index'])

                    if current_constraint['right_index'] < constraint['right_index']: # 分裂当前约束
                        cons_1 = copy.deepcopy(constraint) 
                        cons_1['left'] = current_constraint['right']
                        cons_1['left_index'] = current_constraint['right_index']

                    elif current_constraint['right_index'] > constraint['right_index']:
                        cons_1 = copy.deepcopy(current_constraint)
                        cons_1['left'] = constraint['right']
                        cons_1['left_index'] = constraint['right_index']
                    else:
                        cons_1 = None
                    current_constraint['right'] = constraint['left']
                    current_constraint['right_index'] = constraint['left_index']
                    merged_constraint.append(current_constraint)
                    merged_constraint.append(cons)
                    current_constraint = cons_1

        if (current_constraint is not None) and (current_constraint not in merged_constraint): # 避免最后的约束未添加
            merged_constraint.append(current_constraint)

        return merged_constraint
    
    def _pre_solve(self,solution,cons_type,left_index,right_index):
        if left_index == right_index:
            return solution

        if cons_type in ['Increase', 'Decrease', 'Flat']:
            care_solution = solution[left_index:right_index]
            base_gradient = np.sum(solution[:left_index])
            fore_cumsum = np.cumsum(care_solution) + base_gradient
            back_cumsum = np.cumsum(care_solution[::-1])[::-1] + base_gradient
            if cons_type == 'Increase':
                average_gradient = (np.maximum(fore_cumsum,0) + np.maximum(back_cumsum,0))/2
                solution[left_index] = average_gradient[0] - base_gradient
                solution[left_index+1:right_index] = np.diff(average_gradient)
            elif cons_type == 'Decrease':
                average_gradient = (np.minimum(fore_cumsum,0) + np.minimum(back_cumsum,0))/2
                solution[left_index] = average_gradient[0] - base_gradient
                solution[left_index+1:right_index] = np.diff(average_gradient)
            elif cons_type == 'Flat':
                solution[left_index:right_index] = 0
                if left_index > 0:
                    solution[left_index-1] = solution[left_index-1] - base_gradient

        elif cons_type == 'Convex':
            if left_index <= 0:
                left_index = 1
            solution[left_index+1:right_index] = np.maximum(solution[left_index+1:right_index],0)
        elif cons_type == 'Concave':
            if left_index <= 0:
                left_index = 1
            solution[left_index+1:right_index] = np.minimum(solution[left_index+1:right_index],0)
        elif cons_type == 'Linear':
            solution[left_index] = np.mean(np.cumsum(solution[left_index:right_index]))
            solution[left_index+1:right_index] = 0
        return solution

    
    def analysis_constraints(self,ref_v,solution):
        for cons in self.first_order_constraints_list:
            index_arr = np.searchsorted(ref_v, [cons['left'],cons['right']])
            cons['left_index'] = max(index_arr[0]-1,0) # 左右各延拓一个点，但左闭右开, searchsorted 会自动延展一个点
            cons['right_index'] = min(index_arr[1],len(ref_v) - 1)
        self.first_order_constraints_list = self._merge_constraints(self.first_order_constraints_list,confict_type='Flat')
        
        for cons in self.first_order_constraints_list:
            solution = self._pre_solve(solution,cons['type'],cons['left_index'],cons['right_index'])

        for cons in self.second_order_constraints_list:
            index_arr = np.searchsorted(ref_v, [cons['left'],cons['right']])
            cons['left_index'] = max(index_arr[0]-1,0) # 左右各延拓一个点，但左闭右开
            cons['right_index'] = min(index_arr[1],len(ref_v) - 1)
        
        self.second_order_constraints_list = self._merge_constraints(self.second_order_constraints_list,confict_type='Linear')
        
        for cons in self.second_order_constraints_list:
            solution = self._pre_solve(solution,cons['type'],cons['left_index'],cons['right_index'])

        return solution


    def project_on_constrains(self,current_solution,w,index_list):
        if len(self.first_order_constraints_list) > 0:
            '''
            we form a problem 
                                    min_w ||Ds-w_hat||_2^2
                                    s.t. L(s + xi) >= 0
            '''
            # constraints_list = sorted(self.first_order_constraints_list,key=lambda x:x['left'])
            l = []
            xi = []
            # cumw = np.cumsum(w)
            solution_cum = np.cumsum(current_solution)
            index_in_constraint = []
            last_right = 0
            w_hat_index = []
            local_index = []
            w_hat = []
            w_cum = 0
            D = []
            is_flat = []
            for cons in self.first_order_constraints_list:
                left_index = cons['left_index']
                right_index = cons['right_index']
                index = []
                for idx, w_i in enumerate(w):
                    if (index_list[idx] >= left_index) and (index_list[idx] < right_index):# 不包含右边界
                        if len(local_index) > 0:
                            local_index.append(idx)
                            w_cum += w_i
                            w_hat_index.append(local_index)
                            w_hat.append(w_cum)
                            local_index = []
                            w_cum = 0
                        else:
                            w_hat_index.append([idx])
                            w_hat.append(w_i)
                        index.append(index_list[idx])
                    elif (index_list[idx] >= last_right) and (index_list[idx] < left_index):
                        local_index.append(idx) # 最后一个local_index 不放进 w_hat_index
                        w_cum += w_i
                        
                last_right = right_index
                if len(index) == 0:
                    continue
                index_in_constraint.extend(index)
                N = len(index)
                xi_s = np.zeros(N)
                if cons['type'] == 'Increase':
                    is_flat.append(np.zeros(N))
                    l.append(np.ones(N))
                    for idx in range(N-1):
                        xi_s[idx] = np.min(solution_cum[index[idx]:index[idx+1]])
                    xi_s[N-1] = np.min(solution_cum[index[N-1]:right_index]) # 最后一个点延伸至区域右边界
                elif cons['type'] == 'Decrease':
                    is_flat.append(np.zeros(N))
                    l.append(-np.ones(N))
                    for idx in range(N-1):
                        xi_s[idx] = np.max(solution_cum[index[idx]:index[idx+1]])
                    xi_s[N-1] = np.max(solution_cum[index[N-1]:right_index])
                elif cons['type'] == 'Flat':
                    is_flat.append(np.ones(N))
                    l.append(np.ones(N)) # xi_s = 0
                xi.append(xi_s)

            l = np.concatenate(l)
            xi = np.concatenate(xi)
            is_flat = np.concatenate(is_flat)>0.5


            w_hat = np.array(w_hat)
            diff_v = np.zeros(l.shape[0]+1)
            diff_v[0] = 1
            diff_v[-1] = -1
            D = (circulant(diff_v)[:-1,:-1]).T
            D[is_flat,:] = np.eye(D.shape[0])[is_flat,:] * (1e+10)
            w_his = copy.deepcopy(w_hat)
            w_hat[is_flat] = 0
            
            
            s = self._nn_solve(D,xi,l,w_hat)
            w_new = np.zeros(s.shape[0])
            w_new[0] = s[0]
            w_new[1:] = np.diff(s)
            for idx,local_index in enumerate(w_hat_index):
                err = (w_his[idx] - w_new[idx])/len(local_index)
                for idx_1 in local_index:
                    w[idx_1] = w[idx_1] - err

        if len(self.second_order_constraints_list)>0:
            for cons in self.second_order_constraints_list:
                left_index = cons['left_index']
                right_index = cons['right_index']
                index = [idx for idx in range(len(index_list)) if ((index_list[idx]>left_index) and (index_list[idx]<right_index))] # 约束不包含右边界，左边界点可以为负/正
                if len(index) < 3:
                    continue
                if cons['type'] == 'Convex':
                    w[index] = np.maximum(w[index]+current_solution[index_list[index]],0) - current_solution[index_list[index]]
                elif cons['type'] == 'Concave':
                    w[index] = np.minimum(w[index]+current_solution[index_list[index]],0) - current_solution[index_list[index]]
                elif cons['type'] == 'Linear':
                    w[index] = 0
        
        return w
                
    def _nn_solve(self,D,xi,l,w_hat):
        A = D/l
        b = D.dot(xi) + w_hat
        s_hat = nnls(A,b)[0]
        s = s_hat/l - xi
        return s







class GAM_light:
    def __init__(self, max_iter=100,
                 step_size=1, block_size=50, lambda_1=1,eta=0.001,
                 momentum=0.9,momentum_type='Huber',reg_type='Huber',
                 bin_num=64,randomize=False,verbose=False):
        self.X = None
        self.Y = None
        self.shapeFunctionOptimizerList = None
        self.max_iter = max_iter
        self.step_size = step_size
        self.lambda_1 = lambda_1
        self.block_size = block_size
        self.eta = eta
        self.bin_num = bin_num
        self.verbose = verbose
        self.momentum = momentum
        self.momentum_type = momentum_type
        self.reg_type = reg_type
        self.randomize = randomize
        

    def _init_model(self, X, Y, sample_weights):
        self.X = X
        self.Y = Y  # Original Y is stored
        self.scale_info = {}
        self.scale_info['x_offset'] = np.mean(self.X,axis=0)
        self.scale_info['x_scale'] = np.std(self.X,axis=0)+1e-15
        # self.scale_info['x_offset'] = np.zeros(self.X.shape[1])
        # self.scale_info['x_scale'] = np.ones(self.X.shape[1])
        self.scale_info['y_offset'] = np.mean(self.Y,axis=0)
        self.scale_info['y_scale'] = np.std(self.Y,axis=0)+1e-15
        # self.scale_info['y_offset'] = 0
        # self.scale_info['y_scale'] = 1
        self.X = (self.X - self.scale_info['x_offset'])/(self.scale_info['x_scale'])
        self.Y = (self.Y - self.scale_info['y_offset'])/(self.scale_info['y_scale'])
        if sample_weights is None:
            self.sample_weights = np.ones(self.X.shape[0])
        else:
            self.sample_weights = sample_weights.flatten()
        self.n_features = self.X.shape[1]
        self.fun_dict = np.zeros(self.X.T.shape)
        self.res = self.Y.copy()  # Create a copy instead of a reference
        self.shapeFunctionOptimizerList = []
        for i in range(self.X.shape[1]):
            SFO = shapeFunctionOptimizer(block_size=self.block_size,\
                lambda_1=self.lambda_1, step_size=self.step_size, eta=self.eta, 
                momentum=self.momentum,momentum_type=self.momentum_type,reg_type=self.reg_type,
                bin_num=self.bin_num,randomize=self.randomize)
            SFO._init_model(x=self.X[:,i],sample_weight=self.sample_weights)
            self.shapeFunctionOptimizerList.append(SFO)
    
    def _scale_data(self,x,on='x',idx=None):
        if idx is None:
            return (x-self.scale_info[on+'_offset'])/self.scale_info[on+'_scale']
        else:
            return (x-self.scale_info[on+'_offset'][idx])/self.scale_info[on+'_scale'][idx]
    
    def _rescale_data(self,x,on='x',idx=None):
        if idx is None:
            return x*self.scale_info[on+'_scale'] + self.scale_info[on+'_offset']
        else:
            return x*self.scale_info[on+'_scale'][idx] + self.scale_info[on+'_offset'][idx]
    
        

    def fit(self, X, Y, sample_weights=None,category_features=None,mode='train',max_iter=None):
        if mode == 'train':
            self._init_model(X, Y, sample_weights)
            max_iter = self.max_iter
        elif mode == 'update':
            if max_iter is None:
                max_iter = self.max_iter
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'update'.")
        
        if category_features is None:
            category_features = []
        self.category_features = category_features
        last_train_loss = 1e+10
        for iter in range(max_iter):
            update_seq = np.random.permutation(self.n_features)
            for i in update_seq:
                Regular = (i not in self.category_features)
                filter_result = self.shapeFunctionOptimizerList[i].step(self.res,Regular=Regular)
                self.res -= self.step_size * filter_result

            if ((iter+1) % 50 == 0) and self.verbose:
                train_loss = np.mean(self.res**2)*(self.scale_info['y_scale']**2)
                print(f"iter {iter + 1}: average train_loss={train_loss}")
            if ((iter+1) % 20 == 0):
                train_loss = np.sum(self.sample_weights*(self.res**2))/np.sum(self.sample_weights)
                if abs(train_loss - last_train_loss)/abs(train_loss) < 1e-5:
                    # print(f"Early stop at iter={iter+1}")
                    break
                else:
                    last_train_loss = train_loss
        for ii in range(self.n_features):
            self.shapeFunctionOptimizerList[ii].prepare_prediction()
        if mode == 'train':
            self._calculate_residuals()

    def _create_design_matrix(self, X_scaled, threshold_list):
        """Create design matrix for basis functions representation.
        
        Parameters
        ----------
        X_scaled : array-like
            Scaled input data
        threshold_list : array-like
            Threshold values for the ReLU basis functions
            
        Returns
        -------
        array-like
            Design matrix Z where each row is [1, x, max(x-t1,0), max(x-t2,0), ...]
        """
        n_samples = len(X_scaled)
        n_basis = len(threshold_list) + 2  # +2 for intercept and linear term
        
        # Initialize design matrix
        Z = np.zeros((n_samples, n_basis))
        
        # Set intercept term
        Z[:, 0] = 1
        
        # Set linear term
        Z[:, 1] = X_scaled
        
        # Set ReLU basis terms
        for j, threshold in enumerate(threshold_list):
            Z[:, j+2] = np.maximum(X_scaled - threshold, 0)
        
        return Z

    def _calculate_residuals(self):
        y_pred = self.predict(self.X * self.scale_info['x_scale'] + self.scale_info['x_offset'])
        residuals = (self.Y * self.scale_info['y_scale'] + self.scale_info['y_offset']) - y_pred
        
        # Estimate the residual variance (sigma^2)
        n = len(residuals)
        p = sum([len(sfo.relu_weight) for sfo in self.shapeFunctionOptimizerList])
        self.sigma2 = np.sum(residuals**2) / (n - p)

    
    # def _calculate_variance_covariance_matrix(self,x_scaled, col = 0):
    #     """Calculate the variance-covariance matrix for confidence intervals."""
    #     # Get the residuals
    #     if col>=self.n_features:
    #         raise ValueError(f"Column index {col} is out of range for the number of features.")

        
    #     # Store the design matrices and parameter vectors for each feature
    #     self.design_matrices = []
    #     self.param_vectors = []

    #     sfo = self.shapeFunctionOptimizerList[col]

    #     Z = self._create_design_matrix(x_scaled, sfo.threshold_list)
            
            
            
    #         # Calculate the variance-covariance matrix for this feature
    #     try:
    #         ZtZ_inv = np.linalg.inv(Z.T @ Z)
    #         vcov = ZtZ_inv * self.sigma2
    #     except np.linalg.LinAlgError:
    #         # If matrix is singular, use pseudo-inverse
    #         ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    #         vcov = ZtZ_inv * self.sigma2
    #         return vcov

    def add_constraints(self,constraint_list,idx):
        for cons in constraint_list:
            cons['left'] = self._scale_data(cons['left'],idx=idx)
            cons['right'] = self._scale_data(cons['right'],idx=idx)
        self.shapeFunctionOptimizerList[idx].add_constraints(constraint_list)


    def predict(self, X_test):
        """Make predictions using the fitted GAM model.
        
        Args:
            X_test: numpy array of shape (n_samples, n_features) containing test data
            
        Returns
        -------
        numpy array of shape (n_samples,) containing predictions
        """
        # Get predictions from each shape function
        predictions = self.predict_shape_functions(X_test,intercept=False)
        
        # Sum across features to get final predictions
        return predictions.sum(axis=1)
    
    def get_para(self):
        paras = []
        for ii in range(self.n_features):
            key, value = self.shapeFunctionOptimizerList[ii].get_para()
            para = {}
            para['key'] = key
            para['value'] = value
            paras.append(para)

        return paras
    
    def get_shape_function_data(self, column_name=None, index_set=None, intercept = False):
        """
        Get the data needed for plotting shape functions.
        
        Parameters
        ----------
        column_name : list of str, optional
            Names of the columns/features
        index_set : list of int, optional
            Indices of features to get data for
            
        Returns
        -------
        dict
            Dictionary containing plotting data for each feature:
            {feature_name: {'x': x_values, 'y': y_values, 'density': density_values}}
        """
        if column_name is None:
            if index_set is None:
                column_name = ['feature_'+str(idx) for idx in range(self.n_features)]
            else:
                column_name = ['feature_'+str(idx) for idx in index_set]
        
        if index_set is None:
            index_set = range(self.n_features)
            
        plot_data = {}
        for idx, col_index in enumerate(index_set):
            # Get sorted data for this feature
            sort_index = np.argsort(self.X[:,col_index])
            x_mark = self.X[:,col_index][sort_index]
            y_mark = self.shapeFunctionOptimizerList[col_index].predict(x_mark)
            
            # Rescale the data
            x_mark = self._rescale_data(x_mark, idx=col_index)
            if intercept:
                y_mark = y_mark*self.scale_info['y_scale']
            else:
                y_mark = y_mark*self.scale_info['y_scale']+self.scale_info['y_offset']/self.n_features
            
            # Store the data
            plot_data[column_name[idx]] = {
                'x': x_mark,
                'y': y_mark,
                'density': np.ones(self.X[:,col_index].shape)*(1.5*min(y_mark)-0.5*max(y_mark))
            }
            
        return plot_data

    def plot_shape_functions(self, column_name=None, index_set=None, fig=None, intercept = False):
        """
        Plot the shape functions.
        
        Parameters
        ----------
        column_name : list of str, optional
            Names of the columns/features
        index_set : list of int, optional
            Indices of features to plot
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plots
        """
        # Get the plotting data
        plot_data = self.get_shape_function_data(column_name, index_set, intercept)
        
        # Only create a new figure if one wasn't provided
        if fig is None:
            plt.figure()
        
        # Calculate subplot layout
        n_plots = len(plot_data)
        M = int(round(np.sqrt(n_plots)))
        N = int(np.ceil(np.sqrt(n_plots)))
        
        # Create the plots
        for idx, (feature_name, data) in enumerate(plot_data.items()):
            plt.subplot(M, N, idx+1)
            plt.plot(data['x'], data['y'])
            plt.plot(data['x'], data['density'], 'b+')
            plt.xlabel(feature_name)
            plt.ylabel('score')
        
        plt.tight_layout()
        
        # Return the current figure
        return plt.gcf()

    def predict_shape_functions(self, X_test, intercept = False):
        """Predict individual shape function contributions for each feature.
        
        Args:
            X_test: numpy array of shape (n_samples, n_features) containing test data
            
        Returns
        -------
        numpy array of shape (n_samples, n_features) containing individual shape function predictions
        """
        n_samples, n_features = X_test.shape
        predictions = np.zeros((n_samples, n_features))
        
        scale_info = self.scale_info
        
        for ii in range(n_features):
            # Scale the input data for this feature
            X_scaled = self._scale_data(X_test[:, ii], idx=ii)
            
            # Get raw predictions from shape function optimizer
            feat_pred = self.shapeFunctionOptimizerList[ii].predict(X_scaled)
            
            if intercept:
                # With intercept: just scale the predictions
                predictions[:, ii] = feat_pred * scale_info['y_scale']
            else:
                # Without intercept: scale and add offset divided by n_features
                predictions[:, ii] = (feat_pred * scale_info['y_scale'] + 
                                    scale_info['y_offset'] / n_features)
        
        return predictions

    def get_confidence_intervals(self, X, alpha=0.05):
        """Calculate confidence intervals for shape function predictions.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points at which to evaluate the confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 for 95% CI)
        
        Returns
        -------
        dict
            Dictionary mapping feature indices to tuples of (lower_bound, upper_bound) arrays
        """
        from scipy import stats
        
        confidence_intervals = {}
        n_features = X.shape[1]
        
        # Calculate degrees of freedom (same for all features)
        n = len(self.Y)  # number of observations
        p = len(self.shapeFunctionOptimizerList[0].threshold_list) + 1  # number of parameters
        df = n - p
        
        # Get critical value (same for all features)
        t_value = stats.t.ppf(1 - alpha/2, df)
        
        for i in range(n_features):
            # Scale the input data for this feature
            X_scaled = self._scale_data(X[:, i], idx=i)
            sfo = self.shapeFunctionOptimizerList[i]
            
            # Create design matrix for new points
            Z_new = self._create_design_matrix(X_scaled, sfo.threshold_list)
            
            # Calculate predictions
            predictions = sfo.predict(X_scaled) * self.scale_info['y_scale']
            
            # Calculate standard errors for predictions using stored vcov matrix
            var_pred = np.sum(Z_new * (Z_new @ sfo.vcov), axis=1)
            # 确保方差非负
            var_pred = np.maximum(var_pred, 0)  # 添加这行来处理数值不稳定性
            std_errors = np.sqrt(var_pred)
            
            # Calculate confidence intervals
            margin = t_value * std_errors
            lower = predictions - margin
            upper = predictions + margin
            
            confidence_intervals[i] = (lower, upper)
        
        return confidence_intervals

    def get_shape_function_confidence_intervals(self, alpha=0.05, intercept=False):
        """Calculate confidence intervals for the entire shape functions using basis representation.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals
        intercept : bool, default=False
            Whether to include intercept in calculations
        
        Returns
        -------
        dict
            Dictionary mapping feature indices to tuples of (x_values, lower_bound, upper_bound)
        """
        from scipy import stats
        
        confidence_intervals = {}
        
        # Calculate degrees of freedom
        n = len(self.Y)  # number of observations
        p = len(self.shapeFunctionOptimizerList[0].threshold_list) + 2  # number of parameters (+2 for intercept and linear term)
        df = n - p
        
        # Get critical value
        t_value = stats.t.ppf(1 - alpha/2, df)
        
        # Get shape function data
        shape_data = self.get_shape_function_data(intercept=intercept)
        
        for i in range(self.n_features):
            sfo = self.shapeFunctionOptimizerList[i]
            
            # Get x values from shape function data
            feature_data = shape_data[f'feature_{i}']
            x_values, y_values = feature_data['x'], feature_data['y']
            
            # Scale x values for internal calculations
            x_scaled = self._scale_data(x_values, idx=i)
            
            # Create design matrix for these points using basis functions
            Z = self._create_design_matrix(x_scaled, sfo.threshold_list)
            
            try:
                ZtZ_inv = np.linalg.inv(Z.T @ Z)
                vcov = ZtZ_inv * self.sigma2
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                ZtZ_inv = np.linalg.pinv(Z.T @ Z)
                vcov = ZtZ_inv * self.sigma2
            
            # Calculate standard errors for predictions using basis representation
            var_pred = np.sum(Z * (Z @ vcov), axis=1)
            var_pred = np.maximum(var_pred, 0)  # 添加这行来处理数值不稳定性
            std_errors = np.sqrt(var_pred)
            std_errors *= self.scale_info['y_scale']
            
            
            margin = t_value * std_errors
            lower = y_values - margin
            upper = y_values + margin
            
            
            confidence_intervals[i] = (x_values, lower, upper)
        
        return confidence_intervals







class GAM:
    """
    A wrapper class for GAM_light that provides a more user-friendly interface.
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for training.
    step_size : float, default=1
        Step size for gradient updates.
    block_size : int, default=50
        Block size for optimization.
    lambda_1 : float, default=0.1
        Regularization parameter.
    eta : float, default=0.01
        Learning rate parameter.
    momentum : float, default=0.9
        Momentum parameter for optimization.
    momentum_type : str, default='Huber'
        Type of momentum to use ('Huber' or 'L2').
    reg_type : str, default='Huber'
        Type of regularization to use ('Huber' or 'L2').
    bin_num : int, default=64
        Number of bins for discretizing continuous features.
    randomize : bool, default=False
        Whether to use randomization in optimization.
    verbose : bool, default=False
        Whether to print progress during training.
    """
    
    def __init__(self, feature_prefix='feature_', max_iter=100, step_size=1, block_size=50, 
                 lambda_1=0.1, eta=0.01, momentum=0.9, 
                 momentum_type='Huber', reg_type='Huber',
                 bin_num=64, randomize=False, verbose=False):
        self.model = GAM_light(
            max_iter=max_iter,
            step_size=step_size,
            block_size=block_size,
            lambda_1=lambda_1,
            eta=eta,
            momentum=momentum,
            momentum_type=momentum_type,
            reg_type=reg_type,
            bin_num=bin_num,
            randomize=randomize,
            verbose=verbose
        )
        self.feature_prefix = feature_prefix
        self.feature_names = None
        self.sfo = None  # Will store shape function optimizers after fitting
    
    def fit(self, X, y, sample_weights=None, category_features=None):
        """
        Fit the GAM model to the data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Training data.
        y : array-like
            Target values.
        sample_weights : array-like, optional
            Sample weights.
        category_features : list, optional
            Indices of categorical features.
            
        Returns
        -------
        self : object
            Returns self.
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names = [f'{self.feature_prefix}{i}' for i in range(X.shape[1])]
            print(f'Found no feature names, using {self.feature_prefix} as feature prefix')
        else:
            raise ValueError(f"X must be a pandas DataFrame or numpy array, but got {type(X)}")
        # Fit the model
        self.model.fit(X, y, sample_weights=sample_weights, category_features=category_features, mode='train')
        
        # Store shape function optimizers for later use
        self.sfo = self.model.shapeFunctionOptimizerList
        
        return self
    def update(self, X, y, sample_weights=None, category_features=None):
        """
        Update the model with new data.
        """
        self.model.fit(X, y, sample_weights=sample_weights, category_features=category_features, mode='update')
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Data to predict on.
            
        Returns
        -------
        array-like
            Predicted values.
        """
        if isinstance(X, pd.DataFrame):
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            X = X[self.feature_names].values
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features but got {X.shape[1]}")
        return self.model.predict(X)
    
    def _process_constraint_type(self, constraint_type):
        """Helper method to process constraint type abbreviations."""
        # Map abbreviated constraint types to full names
        constraint_map = {
            'i': 'Increase', 'inc': 'Increase',
            'd': 'Decrease', 'dec': 'Decrease',
            'v': 'Convex', 'vex': 'Convex',
            'c': 'Concave', 'cave': 'Concave'
        }
        
        # Convert abbreviated constraint type to full name if needed
        if isinstance(constraint_type, str) and constraint_type.lower() in constraint_map:
            constraint_type = constraint_map[constraint_type.lower()]
        
        if constraint_type not in ['Increase', 'Decrease', 'Convex', 'Concave']:
            raise ValueError(f"constraint_type must be one of ['Increase', 'Decrease', 'Convex', 'Concave'] or their abbreviations, got {constraint_type}")
        
        return constraint_type
    
    def _resolve_feature_idx(self, feature_idx):
        """Helper method to resolve feature index from name or index."""
        if isinstance(feature_idx, str):
            if feature_idx in self.feature_names:
                return self.feature_names.index(feature_idx)
            else:
                raise ValueError(f"Feature name '{feature_idx}' not found")
        return feature_idx
    
    def _apply_constraints(self, constraints, feature_idx=None):
        """Helper method to apply constraints to features."""
        feature_idx = self._resolve_feature_idx(feature_idx)
        
        if feature_idx is None:
            # Apply to all features
            for i in range(len(self.feature_names)):
                self.model.add_constraints(constraints, i)
        else:
            # Apply to specific feature
            self.model.add_constraints(constraints, feature_idx)
    
    def add_constraint(self, left, right, constraint_type, feature_idx=None):
        """
        Add a single constraint to a specific feature or all features.
        
        Parameters
        ----------
        left : float
            Left boundary of the constraint.
        right : float
            Right boundary of the constraint.
        constraint_type : str
            Type of constraint. Can be one of:
            - 'Increase', 'i', 'inc': Function should be increasing in the range
            - 'Decrease', 'd', 'dec': Function should be decreasing in the range
            - 'Convex', 'v', 'vex': Function should be convex in the range
            - 'Concave', 'c', 'cave': Function should be concave in the range
        feature_idx : int or str, optional
            Index or name of the feature to constrain. If None, apply to all features.
            
        Returns
        -------
        self : object
            Returns self.
        """
        constraint_type = self._process_constraint_type(constraint_type)
        
        constraint = {
            'left': left,
            'right': right,
            'type': constraint_type
        }
        
        self._apply_constraints([constraint], feature_idx)
        return self
    
    def add_constraints(self, constraints, feature_idx=None):
        """
        Add multiple constraints to a specific feature or all features.
        
        Parameters
        ----------
        constraints : list of dict or dict
            If list of dict: Each dict should have 'left', 'right', and 'type' keys.
            If dict: Should have 'left', 'right', and 'type' keys.
            For 'type', you can use abbreviations:
            - 'i', 'inc' for 'Increase'
            - 'd', 'dec' for 'Decrease'
            - 'v', 'vex' for 'Convex'
            - 'c', 'cave' for 'Concave'
        feature_idx : int or str, optional
            Index or name of the feature to constrain. If None, apply to all features.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert single constraint to list
        if isinstance(constraints, dict):
            constraints = [constraints]
            
        # Validate constraints and convert abbreviations
        for constraint in constraints:
            if not all(k in constraint for k in ['left', 'right', 'type']):
                raise ValueError("Each constraint must have 'left', 'right', and 'type' keys")
            
            # Process constraint type
            constraint['type'] = self._process_constraint_type(constraint['type'])
        
        self._apply_constraints(constraints, feature_idx)
        return self
    
    def update_weights(self, left, right, by=1.0, feature_idx=None):
        """
        Update the weights of the shape function in a specific range.
        
        Parameters
        ----------
        left : float
            Left boundary of the range to update.
        right : float
            Right boundary of the range to update.
        by : float, default=1.0
            Amount to scale the weights by in the specified range.
        feature_idx : int or str, optional
            Index or name of the feature to update. If None, apply to all features.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        feature_idx = self._resolve_feature_idx(feature_idx)
        
        # Apply weight updates
        if feature_idx is None:
            # Apply to all features
            features_to_update = range(len(self.feature_names))
        else:
            # Apply to specific feature
            features_to_update = [feature_idx]
        
        for idx in features_to_update:
            # Scale the data to the model's internal scale
            left_scaled = self.model._scale_data(left, idx=idx)
            right_scaled = self.model._scale_data(right, idx=idx)
            
            # Find indices in the threshold list that fall within the range
            sfo = self.model.shapeFunctionOptimizerList[idx]
            mask = (sfo.threshold_list >= left_scaled) & (sfo.threshold_list <= right_scaled)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # Scale the weights
                sfo.relu_weight[indices] *= by
                
                # Recompute the shape function
                sfo.prepare_prediction()
        
        return self
    
    def show(self, data, mode='static', port=8082, waterfall_height="40vh", intercept = False, auto_open = True, ci = True, **kwargs):
        '''
        mode: 'static' or 'interactive'
        port: only used in interactive mode
        waterfall_height: only used in interactive mode. Height of the waterfall component in interactive mode (e.g., "40vh", "300px")
        auto_open: only used in interactive mode. Whether to open the app automatically
        ci: only used in interactive mode. Whether to show the confidence intervals
        '''
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if mode == 'static':
            return plot_static_gam(self, data, **kwargs)
        elif mode == 'interactive':
            # plot_interactive_gam(self, data, **kwargs)
            create_app(self, data, waterfall_height=waterfall_height, port=port, intercept=intercept, auto_open=auto_open, ci = ci)
            # return
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'static' or 'interactive'.")
    def get_confidence_intervals(self, X, alpha=0.05):
        """Calculate confidence intervals for shape function predictions.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Data points at which to evaluate the confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 for 95% CI)
        
        Returns
        -------
        dict
            Dictionary mapping feature names to tuples of (lower_bound, upper_bound) arrays
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Process input X
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
            
        # Get confidence intervals from GAM_light
        ci_dict = self.model.get_confidence_intervals(X, alpha)
        
        # Map feature indices to feature names
        return {self.feature_names[i]: ci_values for i, ci_values in ci_dict.items()}

    def get_shape_functions(self, intercept = False):
        """
        Get the shape functions for all features.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to (x, y) tuples of shape function values.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get shape function data using existing method
        shape_data = self.model.get_shape_function_data(column_name=self.feature_names, intercept=intercept)
        
        # Convert to (x, y) tuples format
        shape_functions = {}
        for feature_name in self.feature_names:
            feature_data = shape_data[feature_name]
            shape_functions[feature_name] = (feature_data['x'], feature_data['y'])
        
        return shape_functions
    

    def get_shape_predictions(self, X, intercept=False):
        """
        Get shape function predictions for each feature at given X values.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Data points at which to evaluate the shape functions.
        intercept : bool, default=False
            Whether to include intercept term in shape function predictions.
            
        Returns
        -------
        dict
            Dictionary mapping feature names to their shape function predictions.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Process input X
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
            
        # Get predictions from underlying model
        predictions = self.model.predict_shape_functions(X, intercept=intercept)
        
        # Map predictions to feature names
        return {name: predictions[:, i] for i, name in enumerate(self.feature_names)} 

    def get_shape_function_confidence_intervals(self, alpha=0.05, intercept=False):
        """Get confidence intervals for the entire shape functions using basis representation.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals
        intercept : bool, default=False
            Whether to include intercept in calculations
        
        Returns
        -------
        dict
            Dictionary mapping feature names to tuples of (x_values, lower_bound, upper_bound)
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get confidence intervals from GAM_light
        ci_dict = self.model.get_shape_function_confidence_intervals(alpha, intercept)
        
        # Map feature indices to feature names
        return {self.feature_names[i]: ci_values for i, ci_values in ci_dict.items()} 