import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq as sp_lstsq
from scipy.linalg import circulant
from scipy.optimize import nnls
import copy
import numba as nb

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
        self.Y = Y
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
        self.res = self.Y
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

    def add_constraints(self,constraint_list,idx):
        for cons in constraint_list:
            cons['left'] = self._scale_data(cons['left'],idx=idx)
            cons['right'] = self._scale_data(cons['right'],idx=idx)
        self.shapeFunctionOptimizerList[idx].add_constraints(constraint_list)


    def predict(self, X_test):
        y_predict = 0
        for ii in range(self.n_features):
            y_predict += self.shapeFunctionOptimizerList[ii].predict(self._scale_data(X_test[:,ii],idx=ii))
        return self._rescale_data(y_predict,on='y')
    
    def get_para(self):
        paras = []
        for ii in range(self.n_features):
            key, value = self.shapeFunctionOptimizerList[ii].get_para()
            para = {}
            para['key'] = key
            para['value'] = value
            paras.append(para)

        return paras
    
    def plot_shape_functions(self, column_name=None, index_set=None, fig=None):
        if column_name is None:
            if index_set is None:
                column_name = ['feature_'+str(idx) for idx in range(self.n_features)]
            else:
                column_name = ['feature_'+str(idx) for idx in index_set]
        
        # Only create a new figure if one wasn't provided
        if fig is None:
            plt.figure()
        
        if index_set is None:
            index_set = range(self.n_features)
        M = int(round(np.sqrt(len(index_set))))
        N = int(np.ceil(np.sqrt(len(index_set))))
        for idx,col_index in enumerate(index_set):
            sort_index = np.argsort(self.X[:,col_index])
            plt.subplot(M,N,idx+1)
            x_mark = self.X[:,col_index][sort_index]
            y_mark = self.shapeFunctionOptimizerList[col_index].predict(x_mark)
            x_mark = self._rescale_data(x_mark,idx=col_index)
            y_mark = y_mark*self.scale_info['y_scale']+self.scale_info['y_offset']/self.n_features
            plt.plot(x_mark,y_mark)
            baseline_min = min(y_mark)
            baseline_max = max(y_mark)
            plt.plot(x_mark,np.ones(self.X[:,col_index].shape)*(1.5*baseline_min-0.5*baseline_max),'b+')
            plt.xlabel(column_name[idx])
            plt.ylabel('score')
        
        plt.tight_layout()
        
        # Return the current figure
        return plt.gcf()


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
            X = X.values
        
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
    
    def show(self, feature_indices=None, figsize=(12, 10), display=True, 
             title=None, xlabel=None, ylabel="Attribution", show_density=True, 
             color='#1f77b4', linestyle='-', linewidth=2, alpha=0.7, 
             density_color='#ff7f0e', density_alpha=0.3, density_markersize=5,
             use_color_cycle=False, save_path=None, dpi=300, xlim=None, ylim=None, 
             layout=None, **kwargs):
        """
        Plot the shape functions for the specified features.
        
        Parameters
        ----------
        feature_indices : int, str, or list, optional
            Indices or names of features to plot. If None, all features are plotted.
            Can be a single integer index, a single string feature name, or a list 
            containing a mix of integer indices and string feature names.
        figsize : tuple, default=(12, 10)
            Figure size.
        display : bool, default=True
            Whether to display the figure immediately using plt.show().
        title : str or list of str, optional
            Title for the plot or list of titles for each subplot.
        xlabel : str or list of str, optional
            Label for x-axis or list of labels for each subplot. If None, feature names are used.
        ylabel : str, default="Attribution"
            Label for y-axis.
        show_density : bool, default=True
            Whether to show density of data points as a rug plot.
        color : str or list, default='#1f77b4'
            Color for the line or list of colors for each subplot. By default, all plots use the same blue color.
        linestyle : str or list, default='-'
            Line style or list of line styles for each subplot.
        linewidth : float or list, default=2
            Line width or list of line widths for each subplot.
        alpha : float, default=0.7
            Alpha transparency for the line plot.
        density_color : str or list, default='#ff7f0e'
            Color for the density plot. By default, all density plots use the same orange color.
        density_alpha : float, default=0.3
            Alpha transparency for the density plot.
        density_markersize : float, default=5
            Size of markers in the density plot.
        use_color_cycle : bool, default=False
            If True, uses matplotlib's default color cycle for lines instead of a single color.
        save_path : str, optional
            Path to save the figure. If provided, the figure will be saved to this location.
            The file format is determined by the file extension (e.g., .png, .pdf, .svg).
        dpi : int, default=300
            Resolution of the saved figure in dots per inch.
        xlim : tuple or list of tuples, optional
            The x limits (min, max) for the plot or a list of tuples for each subplot.
        ylim : tuple or list of tuples, optional
            The y limits (min, max) for the plot or a list of tuples for each subplot.
        layout : tuple, optional
            The layout of subplots as (rows, cols). If None, a square-ish layout is used.
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Process feature indices
        if feature_indices is None:
            # Use all features
            indices = list(range(len(self.feature_names)))
        else:
            # Convert to list if a single index/name is provided
            if isinstance(feature_indices, (int, str)):
                feature_indices = [feature_indices]
            
            # Convert any feature names to indices
            indices = []
            for idx in feature_indices:
                if isinstance(idx, str):
                    if idx in self.feature_names:
                        indices.append(self.feature_names.index(idx))
                elif isinstance(idx, int):
                    if 0 <= idx < len(self.feature_names):
                        indices.append(idx)
                else:
                    raise ValueError(f"Feature identifier must be a string or integer, got {type(idx)}")
        
        # Create a figure with the specified size
        fig = plt.figure(figsize=figsize)
        
        # Prepare plot parameters
        n_plots = len(indices)
        
        # Determine subplot layout
        if layout is None:
            # Default to square-ish layout
            M = int(round(np.sqrt(n_plots)))
            N = int(np.ceil(n_plots / M))
        else:
            # Use specified layout
            M, N = layout
            if M * N < n_plots:
                print(f"Warning: Layout {layout} can only fit {M*N} plots, but {n_plots} were requested.")
                # Adjust to fit all plots
                N = int(np.ceil(n_plots / M))
        
        # Handle list or single value for plot parameters
        def ensure_list(param, n):
            if isinstance(param, list):
                return param
            else:
                return [param] * n
        
        # Use matplotlib's default color cycle if requested
        if use_color_cycle:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            # Repeat the color cycle if needed
            colors = [colors[i % len(colors)] for i in range(n_plots)]
        else:
            colors = ensure_list(color, n_plots)
        
        linestyles = ensure_list(linestyle, n_plots)
        linewidths = ensure_list(linewidth, n_plots)
        
        # Handle density colors
        density_colors = ensure_list(density_color, n_plots)
        
        # Handle titles and xlabels
        if title is not None:
            titles = ensure_list(title, n_plots)
        else:
            titles = [None] * n_plots
        
        if xlabel is not None:
            xlabels = ensure_list(xlabel, n_plots)
        else:
            xlabels = [self.feature_names[i] for i in indices]
        
        # Handle axis limits
        if xlim is not None:
            xlims = ensure_list(xlim, n_plots)
        else:
            xlims = [None] * n_plots
        
        if ylim is not None:
            ylims = ensure_list(ylim, n_plots)
        else:
            ylims = [None] * n_plots
        
        # Create subplots
        for i, (idx, col_index) in enumerate(enumerate(indices)):
            if i < M * N:  # Only create plots that fit in the layout
                ax = plt.subplot(M, N, i+1)
                
                # Get data for the feature
                sort_index = np.argsort(self.model.X[:, col_index])
                x_mark = self.model.X[:, col_index][sort_index]
                y_mark = self.model.shapeFunctionOptimizerList[col_index].predict(x_mark)
                
                # Rescale data
                x_mark = self.model._rescale_data(x_mark, idx=col_index)
                y_mark = y_mark * self.model.scale_info['y_scale'] + self.model.scale_info['y_offset'] / len(self.sfo)
                
                # Plot shape function
                ax.plot(x_mark, y_mark, color=colors[i], linestyle=linestyles[i], 
                        linewidth=linewidths[i], alpha=alpha, **kwargs)
                
                # Show density if requested
                if show_density:
                    # Get original data for density plot
                    x_orig = self.model._rescale_data(self.model.X[:, col_index], idx=col_index)
                    
                    # Add rug plot at the bottom
                    baseline = min(y_mark) - 0.1 * (max(y_mark) - min(y_mark))
                    ax.plot(x_orig, np.ones_like(x_orig) * baseline, '|', 
                            color=density_colors[i], alpha=density_alpha, markersize=density_markersize)
                
                # Set labels and title
                ax.set_xlabel(xlabels[i])
                ax.set_ylabel(ylabel)
                if titles[i]:
                    ax.set_title(titles[i])
                
                # Set axis limits if provided
                if xlims[i] is not None:
                    ax.set_xlim(xlims[i])
                if ylims[i] is not None:
                    ax.set_ylim(ylims[i])
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        # Display the figure if requested
        if display:
            plt.show()
        
        # Return the figure
        return fig

    def get_shape_functions(self):
        """
        Get the shape functions for all features.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to (x, y) tuples of shape function values.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        shape_functions = {}
        for i, feature_name in enumerate(self.feature_names):
            x_values, y_values = self.model.shapeFunctionOptimizerList[i].get_para()
            
            # Rescale the values back to original scale
            x_rescaled = self.model._rescale_data(x_values, idx=i)
            y_rescaled = y_values * self.model.scale_info['y_scale'] + self.model.scale_info['y_offset'] / len(self.sfo)
            
            shape_functions[feature_name] = (x_rescaled, y_rescaled)
        
        return shape_functions
    
    def analyze_feature_importance(self):
        """
        Analyze the importance of each feature based on the range of its shape function.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with feature names and their importance scores.
        """
        if self.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        importances = []
        for i, feature_name in enumerate(self.feature_names):
            _, y_values = self.model.shapeFunctionOptimizerList[i].get_para()
            y_rescaled = y_values * self.model.scale_info['y_scale']
            importance = np.max(y_rescaled) - np.min(y_rescaled)
            importances.append(importance)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        return importance_df.sort_values('Importance', ascending=False) 