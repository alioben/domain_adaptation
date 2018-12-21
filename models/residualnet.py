import tensorflow as tf
from helpers import *

class ResidualNet:
       
    ''' NETWORK INITIALISATION '''
    
    def __init__(self, max_l=32, max_r=32):
        self.weights = {}
        self.wT = None
        self.max_l = max_l
        self.max_r = max_r
        self.src_stream = None
    

    def build(self, src_stream):
        ''' Returns the weights of the target stream from
        the weights of the source and residual streams '''
        self.src_stream=src_stream
        for name, var in src_stream.get_weights(keys=True):
            self.create_residual_weights(name)
        return self
    

    
    ''' RESIDUAL LOSS '''
          
    def get_resid_loss(self):
        ''' Returns the residual stream loss '''
        resid_loss = l2_loss_list(self.get_target_weights(add_weight=False).values())
        resid_loss = resid_loss-tf.log(resid_loss)          
        return resid_loss
    
    
        
    ''' PROXIMAL GRAIDENT DESCENT '''
    
    def get_A_solution_weight(self, name, adam_lr, lambda_p):
        ''' Computes the solutions for A1 and A2 '''
        T_hat = self.residual_transform_weight(name, ret_in_sig=True)
        T = self.project_T(T_hat, adam_lr, lambda_p)
        A1_right = tf.transpose(tf.concat([tf.transpose(self.get_weight(name, 'A1')), T-self.get_weight(name, 'D')], axis=1))
        eye_shape_A1 = self.get_src_var(name).get_shape().as_list()[0]
        A1_left = tf.transpose(tf.concat([tf.eye(eye_shape_A1), tf.matmul(self.get_src_var(name), self.get_weight(name, 'A2'))], axis=1))
        A1 = tf.matrix_solve_ls(A1_left, A1_right, l2_regularizer=1e-5, fast=False)

        # eye_shape_A2 = A1.get_shape().as_list()[1]
        # A2_left = tf.concat([tf.eye(eye_shape_A2), tf.matmul(tf.transpose(A1), wS['fc2'])], axis=1)
        # A2_right = tf.concat([tf.transpose(wR['A2_4']), T-wR['D_4']], axis=1)
        A2_left = tf.matmul(tf.transpose(A1), self.get_src_var(name))
        A2_right = T-self.get_weight(name, 'D')
        A2 = tf.matrix_solve_ls(A2_left, A2_right, l2_regularizer=1e-5, fast=False)
       
        return A1, A2
    
    def update_A(self, adam_lr, lambda_p):
        ''' Returns the solution for A for all layers '''
        ops = []
        for name, var in self.src_stream.get_weights(keys=True):
            new_A1, new_A2 = self.get_A_solution_weight(name, adam_lr, lambda_p)
            ops.append(tf.assign(self.get_weight(name, 'A1'), new_A1))
            ops.append(tf.assign(self.get_weight(name, 'A2'), new_A2))
        return ops
        
    def project_T(self, T_hat, t, lambda_):
        ''' Computes the projection of T_hat by zeroing
            out the columns and rows with small l2 loss '''
        h,w = T_hat.get_shape().as_list()

        # Zero out ineffective cols
        P = tf.cast(h, tf.float32)
        l2_loss = tf.reshape(tf.reduce_sum(tf.square(T_hat), 0), [1, -1])
        mask = tf.tile(tf.nn.relu(1-(t*lambda_*tf.sqrt(P))*1./l2_loss), [h, 1])
        T = tf.multiply(mask, T_hat)

        # Zero out ineffective rows
        P = tf.cast(w, tf.float32)
        l2_loss = tf.reshape(tf.reduce_sum(tf.square(T), 1), [-1, 1])
        mask = tf.tile(tf.nn.relu(1-(t*lambda_*tf.sqrt(P))*1./l2_loss), [1, w])
        T = tf.multiply(mask, T)

        return T
    
    
    
    ''' RESIDUAL TRANSFORMATION OF SOURCE WEIGHTS '''
                    
    def residual_transform(self,W,A1,A2,B1,B2,D,add_weight=True,ret_in_sig=False):
        ''' Converts a source stream weight into a target stream weight '''
        in_sigmoid = tf.add(tf.matmul(tf.matmul(tf.transpose(A1), W), A2), D)
        if ret_in_sig:
            return in_sigmoid
        in_sigmoid = tf.nn.sigmoid(in_sigmoid)
        target_weight = tf.matmul(tf.matmul(B1, tf.nn.sigmoid(in_sigmoid)), tf.transpose(B2))
        if not add_weight:
            return target_weight
        target_weight = tf.add(target_weight, W)
        return target_weight
    
    def residual_transform_weight(self,name,add_weight=True,ret_in_sig=False):
        ''' Returns the target weights of the corresponding source stream '''
        return self.residual_transform(self.get_src_var(name), 
                                       self.weights[name+"_A1"],
                                       self.weights[name+"_A2"],
                                       self.weights[name+"_B1"],
                                       self.weights[name+"_B2"],
                                       self.weights[name+"_D"],
                                       add_weight,ret_in_sig)
    
    
    
    ''' HELPER FUNCTIONS '''
        
    def get_weight(self, name, subname):
        ''' Returns the variable corresponding to the weight '''
        return self.weights[name+'_'+subname]
    
    def get_weights(self, keys=False):
        if keys:
            return self.weights
        return list(self.weights.values())
    
    def get_target_weights(self, add_weight=True, reshape=True):
        ''' Return the weights for target stream '''
        wT = {}
        print('------ Target Weights -------')
        for name, var in self.src_stream.get_weights(keys=True):
            wT[name] = self.residual_transform_weight(name, add_weight=add_weight)
            if reshape:
                wT[name] = tf.reshape( wT[name], var.get_shape().as_list() )
            print('{}\t{}'.format( name, wT[name].get_shape().as_list() ))
        return wT
    
    def create_residual_weights(self, name):
        ''' Creates the residual weights '''
        h,w = self.get_src_var(name).get_shape().as_list()
        self.weights[name+"_A1"] = self.create_variable([h, self.max_l], name=name+"_A1")
        self.weights[name+"_B1"] = self.create_variable([h, self.max_l], name=name+"_B1")
        self.weights[name+"_A2"] = self.create_variable([w, self.max_r], name=name+"_A2")
        self.weights[name+"_B2"] = self.create_variable([w, self.max_r], name=name+"_B2")
        self.weights[name+"_D"] = self.create_variable([self.max_l, self.max_r], name=name+"_D")
    
    def create_variable(self, size, name=None):
        ''' Creates a new residual variable and initialize its weights '''
        return tf.Variable(random_uniform(size, 1e-6, 1e-5), name=name)
    
    def get_src_var(self, name):
        ''' Returns a source weight with appropriate weights '''
        var = self.src_stream.get_weight(name)
        shape = var.get_shape().as_list()
        if len(shape) > 2:
            k1,k2,indim,outdim = shape
            h,w=k1*k2*indim, outdim
            return tf.reshape(var, [h,w])
        return var