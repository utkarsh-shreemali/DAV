import numpy as np

def feature_sign_search(A , y, gamma):
    x=np.zeros(A.shape[1])
    sign=np.zeros(A.shape[1], dtype=np.int8)
    active_set=set()
    
    A_transpose_A=np.dot(A.T,A)
    A_transpose_y=np.dot(A.T,y)
    
    func_grad=-2*A_transpose_y + 2*np.dot(A_transpose_A,x)
    
    conditionA_value=np.inf
    conditionB_value=0
    
    while conditionA_value>gamma or not np.allclose(conditionB_value,0):
        if np.allclose(conditionB_value,0):
            i=np.argmax(np.abs(func_grad)*(sign==0))
            if func_grad[i]>gamma:
                sign[i]=-1
                x[i]=0
                active_set.add(i)
            elif func_grad[i]<-gamma:
                sign[i]=1
                x[i]=0
                active_set.add(i)
            if len(active_set)==0:
                break
        columns_to_consider=np.array(sorted(active_set))
        #print(columns_to_consider)
        Ahat_transpose_Ahat=A_transpose_A[np.ix_(columns_to_consider,columns_to_consider)]
        Ahat_transpose_yhat=A_transpose_y[columns_to_consider]
        signhat=sign[columns_to_consider]
        #xhat=x[columns_to_consider]
        
        #Ahat_transpose_Ahat_inverse = np.linalg.inv(Ahat_transpose_Ahat)
        if np.linalg.det(Ahat_transpose_Ahat) == 0:
            x_new=np.linalg.solve(np.atleast_2d(Ahat_transpose_Ahat+0.001*np.eye(Ahat_transpose_Ahat.shape[0])),(Ahat_transpose_yhat - gamma * signhat/2))
        else:
            x_new=np.linalg.solve(np.atleast_2d(Ahat_transpose_Ahat),(Ahat_transpose_yhat - gamma * signhat/2))
        #print(x_new)
        sign_new=np.sign(x_new)
        #print(sign_new)
        x_old=x[columns_to_consider]
        sign_change_index=np.where(abs(sign_new - signhat)==2)[0]
        #print(sign_change_index)
        if len(sign_change_index)>0:
            optimal_obj=np.inf
            optimal_x=None
            optimal_x=x_new
            
            optimal_obj=(np.dot(y.T,y)+(np.dot(x_new,np.dot(Ahat_transpose_Ahat,x_new)) - 2*np.dot(x_new,Ahat_transpose_yhat))+gamma*abs(x_new).sum())
            
            for j in sign_change_index:
                px=x_new[j]
                py=x_old[j]
                slope=py/(py-px)
                x_curr=x_old - slope * (x_old - x_new)
                curr_obj = np.dot(y.T,y) + (np.dot(x_curr,np.dot(Ahat_transpose_Ahat,x_curr)) - 2 * np.dot(x_curr,Ahat_transpose_yhat ) + gamma * abs(x_curr).sum())
                
                if curr_obj < optimal_obj:
                    optimal_obj = curr_obj
                    #optimal_slope = slope
                    optimal_x = x_curr
        else:
            optimal_x=x_new
        x[columns_to_consider]=optimal_x
        zero_coefficients = columns_to_consider[np.abs(x[columns_to_consider]) < 1e-16]
        x[zero_coefficients] = 0
        sign[columns_to_consider]=np.int8(np.sign(x[columns_to_consider]))
        active_set.difference_update(zero_coefficients)
        func_grad = -2* A_transpose_y +2*np.dot(A_transpose_A,x)
        conditionA_value=np.max(abs(func_grad[sign==0]))
        conditionB_value=np.max(abs(func_grad[sign!=0] + gamma * sign[sign!=0]))
    return x
        
        
        
        
    
    
    